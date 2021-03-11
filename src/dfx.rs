extern crate test;

use super::discretize::Discretizer;
use super::distances;
use super::pgres;

use crossbeam::crossbeam_channel::bounded;
use num_format::{Locale, ToFormattedString};
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use threadpool::ThreadPool;

const NUM_WORKERS: usize = 14;
const CHANNEL_BUFFER: usize = NUM_WORKERS * 20;
const INSERT_BATCH_SIZE: usize = 1000;

fn vector_inserter(
    r: &crossbeam::crossbeam_channel::Receiver<pgres::Add_Request>,
    w: &crossbeam::crossbeam_channel::Sender<pgres::Add_Request>,
    pg_helper: &pgres::PG_Helper,
    NUM_BUCKETS: usize,
) {
    let mut existing_vectors = HashMap::new();
    let mut batch_vectors = HashMap::new();
    let disc = Discretizer::New("mxmn.json".to_string(), NUM_BUCKETS as usize);

    {
        let old_vectors = pg_helper.Get_Vecs();
        for vector_data in old_vectors {
            existing_vectors.insert(vector_data.vec, vector_data.id);
        }
    } // old_vectors dropped

    let mut batch = Vec::new();
    for mut x in r.iter() {
        x.vec = disc.Discretize(&x.embedding);
        if existing_vectors.contains_key(&x.vec) {
            x.id = *existing_vectors.get(&x.vec).unwrap();
        } else if !batch_vectors.contains_key(&x.vec) {
            batch_vectors.insert(x.vec.clone(), Vec::new());
            batch.push(x);
        } else {
            batch_vectors.get_mut(&x.vec).unwrap().push(x);
        }

        // I am not sure if is empty will fire on the last insert here...
        // The intent is that if there is nothing else to do, to the insert.
        if (batch.len() + batch_vectors.len() > INSERT_BATCH_SIZE)
            || (r.is_empty() && batch.len() + batch_vectors.len() > 0)
        {
            pg_helper.insert_vectors(&mut batch);
            for ar in batch {
                existing_vectors.insert(ar.vec.clone(), ar.id);
                if let Err(e) = w.send(ar) {
                    println!("{}", e);
                }
            }
            for (vec, array) in batch_vectors {
                for mut ar in array {
                    ar.id = *existing_vectors.get(&vec).unwrap();
                    w.send(ar).unwrap();
                }
            }
            batch = Vec::new();
            batch_vectors = HashMap::new();
        }
    }
}

fn phrase_inserter(
    r: &crossbeam::crossbeam_channel::Receiver<pgres::Add_Request>,
    pg_helper: &pgres::PG_Helper,
) {
    let mut batch = Vec::new();
    for x in r.iter() {
        batch.push(x);
        if batch.len() > INSERT_BATCH_SIZE || (r.is_empty() && batch.len() > 0) {
            pg_helper.insert_phrases(&batch);
            batch = Vec::new();
        }
    }
    // this wont trip until the channel is closed
    if batch.len() > 0 {
        pg_helper.insert_phrases(&batch);
    }
}

#[derive(Clone)]
pub struct Vec_Search {
    pub chan_in: crossbeam::crossbeam_channel::Sender<pgres::Add_Request>,
    disc: Discretizer,
    add_pool: threadpool::ThreadPool,
    query_pool: threadpool::ThreadPool,
    pg_helper: pgres::PG_Helper,
    mode: String,
    search_vectors: Arc<Vec<pgres::Vector_Data>>,
    partition_size: usize,
    remainder: usize,
}

use deepsize::DeepSizeOf;

impl Vec_Search {
    pub fn New(bucket_count: usize, mode: String) -> Vec_Search {
        let disc = Discretizer::New("mxmn.json".to_string(), bucket_count as usize);
        let mut pg_helper = pgres::PG_Helper::New();

        let (vector_insert_send, vector_insert_read) = bounded(CHANNEL_BUFFER);
        let (phrase_insert_send, phrase_insert_read) = bounded(CHANNEL_BUFFER);
        let add_pool = ThreadPool::new(2);
        let query_pool = ThreadPool::new(NUM_WORKERS);

        // build the query bundler....
        let partition_size;
        let remainder;
        let search_vectors;
        if mode == "add" {
            println!("Loading add mode...");
            partition_size = 0;
            remainder = 0;
            search_vectors = Arc::new(Vec::new());
            let mut pg_clone = pg_helper.clone();

            add_pool.execute(move || {
                vector_inserter(
                    &vector_insert_read,
                    &phrase_insert_send,
                    &mut pg_clone,
                    bucket_count,
                )
            });
            let mut pg_clone = pg_helper.clone();

            add_pool.execute(move || phrase_inserter(&phrase_insert_read, &mut pg_clone));
        } else {
            let now = Instant::now();
            println!("Loading query mode...");
            let old_vectors = pg_helper.Get_Vecs();
            drop(pg_helper);
            pg_helper = pgres::PG_Helper::New();
            partition_size = old_vectors.len() / NUM_WORKERS;
            remainder = old_vectors.len() - NUM_WORKERS * partition_size;
            search_vectors = Arc::new(old_vectors);
            println!(
                "Loaded {} Vectors for Querying in {:.3} seconds, of size {} bytes",
                search_vectors.len().to_formatted_string(&Locale::en),
                now.elapsed().as_millis() as f32 / 1000.0,
                (search_vectors.capacity() * search_vectors[0].deep_size_of()
                    + mem::size_of_val(&(1 as usize)) * 2)
                    .to_formatted_string(&Locale::en),
            );
        }

        return Vec_Search {
            chan_in: vector_insert_send,
            add_pool,
            query_pool,
            disc,
            pg_helper,
            mode: "add".to_string(),
            search_vectors,
            partition_size,
            remainder,
        };
    }

    pub fn Encode(&self, a: &[f32]) -> Vec<u64> {
        return self.disc.Discretize(a);
    }

    pub fn Add(
        &self,
        phrase: String,
        embedding: Vec<f32>,
        class: i32,
    ) -> Result<(), crossbeam::SendError<pgres::Add_Request>> {
        let ar = pgres::Add_Request {
            embedding,
            phrase,
            class,
            vec: Vec::new(),
            id: -1,
        };
        self.chan_in.send(ar)
    }

    pub fn KNN(&self, v_in: Vec<u64>, k: usize) -> Vec<pgres::result> {
        let result_batch = Arc::new(Mutex::new(Vec::new()));

        for i in 0..NUM_WORKERS {
            let mut end = self.partition_size * (i + 1);
            if i == NUM_WORKERS - 1 {
                end += self.remainder;
            }
            let encoded_clone = v_in.clone();
            let results_clone = result_batch.clone();
            let partition_size = self.partition_size;
            let vecref = self.search_vectors.clone();

            self.query_pool.execute(move || {
                let mut res = get_results(&encoded_clone, &vecref, partition_size * i, end, k);
                {
                    let mut data = results_clone.lock().unwrap();
                    data.append(&mut res);
                }
            });
        }

        self.query_pool.join();

        let mut results = Arc::try_unwrap(result_batch).unwrap().into_inner().unwrap();
        results.sort();
        results.drain(k..);
        results
    }

    pub fn Query(&self, v_in: Vec<f32>, len: usize, k: usize, boost: f32) -> Vec<pgres::result> {
        let total_q_time = Instant::now();
        let encoded = self.Encode(&v_in);
        let mut results = self.KNN(encoded, k);
        self.pg_helper.Get_Phrases(&mut results);

        println!(
            "DFX Query Completed in {:.3} seconds",
            total_q_time.elapsed().as_millis() as f32 / 1000.0
        );

        results.to_vec()
    }
}

pub unsafe fn Compare(a: &[u64], b: &[u64]) -> u32 {
    return distances::distances(a, b);
}

fn get_results<'a>(
    v_in: &Vec<u64>,
    vectors_to_search: &Vec<pgres::Vector_Data>,
    slice_start: usize,
    slice_end: usize,
    k: usize,
) -> Vec<pgres::result> {
    let mut results = Vec::new();

    for vector_data in vectors_to_search[slice_start..slice_end].iter() {
        let sco: u32;
        unsafe {
            sco = Compare(&v_in, &vector_data.vec);
        }

        if results.len() < k {
            results.push(pgres::result {
                score: sco,
                vec_id: vector_data.id,
                phrases: Vec::new(),
            });
            results.sort();
        } else if sco < results[k - 1].score {
            results[k - 1] = pgres::result {
                score: sco,
                vec_id: vector_data.id,
                phrases: Vec::new(),
            };
            results.sort();
        }
    }
    return results;
}
