use deepsize::DeepSizeOf;
use itertools::Itertools;
use postgres::fallible_iterator::FallibleIterator;
use postgres::{Client, NoTls};
use serde::{Deserialize, Serialize};
use sql_builder::quote;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::iter;
use std::sync::{Arc, Mutex};
use tokio_postgres::types::ToSql;

#[derive(Clone)]
pub struct PG_Helper {
    add_lock: Arc<Mutex<bool>>,
}

#[derive(Clone, DeepSizeOf)]
pub struct Vector_Data {
    pub id: i32,
    pub vec: Vec<u64>,
    pub phrase_len: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
pub struct Add_Request {
    pub embedding: Vec<f32>,
    pub phrase: String,
    pub class: i32,
    pub vec: Vec<u64>,
    pub id: i32,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct Phrase_Info {
    pub phrase: String,
    pub class: i32,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct result {
    pub score: u32,
    pub vec_id: i32,
    pub phrases: Vec<Phrase_Info>,
}

impl Ord for result {
    fn cmp(&self, other: &result) -> Ordering {
        // other.score.cmp(&self.score);
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for result {
    fn partial_cmp(&self, other: &result) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
pub fn vec_to_string(vec_in: &Vec<u64>) -> String {
    vec_in.into_iter().map(|x| x.to_string()).join("|")
}

fn vec_from_string(str_in: String) -> Vec<u64> {
    str_in
        .split("|")
        .map(|x| x.parse::<u64>().unwrap())
        .collect_vec()
}

impl PG_Helper {
    pub fn New() -> PG_Helper {
        let me = PG_Helper {
            add_lock: Arc::new(Mutex::new(true)),
        };

        let mut client = me.get_client();

        let t = client.query_one("SELECT max(id) from Vectors;", &[]);
        if t.is_err() {
            me.Initialize_DB();
        }
        return me;
    }

    fn get_client(&self) -> postgres::Client {
        let mut client = Client::connect(
            "host=172.17.0.1 user=postgres password=OMG_Vectors_Vectors_Vectors!",
            NoTls,
        )
        .unwrap();
        return client;
    }

    pub fn Initialize_DB(&self) {
        let mut client = self.get_client();

        let q = "CREATE UNLOGGED TABLE IF NOT EXISTS Vectors (id SERIAL, phrase_len INT, Data TEXT) WITH (autovacuum_enabled=false) ";

        client.execute(&q[..], &[]).unwrap();

        client
            .execute(
                "CREATE  UNLOGGED TABLE IF NOT EXISTS Phrases  (
                vid INT,
                class INT,
                phrase TEXT
                ) WITH (autovacuum_enabled=false) ;",
                &[],
            )
            .unwrap();

        client
            .execute(
                "
                    CREATE INDEX IF NOT EXISTS phrase_idx ON Phrases USING HASH (vid);
                ",
                &[],
            )
            .unwrap();
    }

    pub fn insert_phrase(&mut self, vid: i32, phrase: String, class: i32) {
        {
            let q = format!(
                "INSERT INTO Phrases (vid, phrase, class) VALUES ({}, $1, {});",
                vid, class
            );
            let mut client = self.get_client();
            client.execute(&q[..], &[&phrase]).unwrap();
        }
    }

    pub fn insert_phrases(&self, inputs: &Vec<Add_Request>) {
        let mut client = self.get_client();
        let mut strings = Vec::new();
        for ar in inputs {
            strings.push(format!("({},{},{})", ar.id, ar.class, quote(&ar.phrase)));
        }

        let q = format!(
            "INSERT INTO Phrases(vid, class, phrase) VALUES {}",
            strings.join(",")
        );

        client.execute(&q[..], &[]).unwrap();
    }

    pub fn insert_vector(&self, v: &Vec<u64>, phrase_len: u64) -> i32 {
        let vec_string = vec_to_string(v);

        let q = format!( "INSERT INTO Vectors  (phrase_len, id, Data) VALUES ({}, DEFAULT, '{}') RETURNING id as ID;",
            phrase_len,
        vec_string,
        );

        let mut client = self.get_client();
        match client.query_one(&q[..], &[]) {
            Ok(id) => id.get::<'_, _, i32>("id"),
            Err(_e) => {
                let q = format!("SELECT id from Vectors WHERE data = '{}';", vec_string,);
                client.query_one(&q[..], &[]).unwrap().get("id")
            }
        }
    }

    pub fn insert_vectors(&self, inputs: &mut Vec<Add_Request>) {
        let mut client = self.get_client();
        let mut strings = Vec::new();
        for ar in inputs.iter().by_ref() {
            strings.push(format!(
                "(DEFAULT,{},'{}')",
                ar.phrase.len(),
                vec_to_string(&ar.vec)
            ));
        }
        let q = format!(
            "INSERT INTO Vectors(id, phrase_len, data) VALUES {} RETURNING id",
            strings.join(",")
        );
        for (row, ar) in client
            .query(&q[..], &[])
            .unwrap()
            .iter()
            .zip(inputs.iter_mut())
        {
            // println!("Row: {:?}", row);
            ar.id = row.get("id");
        }
    }

    pub fn Get_Vecs(&self) -> Vec<Vector_Data> {
        let mut vectors: Vec<Vector_Data> = Vec::new();
        let mut client = self.get_client();
        let params: Vec<String> = vec![];

        let mut ri = client
            .query_raw(
                "SELECT Data, phrase_len, id from Vectors;",
                params.iter().map(|p| p as &dyn ToSql),
            )
            .unwrap();
        while let Some(row) = ri.next().unwrap() {
            vectors.push(Vector_Data {
                id: row.get::<'_, _, i32>("id"),
                phrase_len: row.get::<'_, _, i32>("phrase_len") as usize,
                vec: vec_from_string(row.get("Data")),
            });
        }
        return vectors;
    }

    pub fn Get_Phrases(&self, results: &mut Vec<result>) {
        let mut phrase_set = HashSet::new();
        let mut client = self.get_client();
        for resu in results.iter_mut() {
            let q = format!(
                "SELECT phrase, class from Phrases where vid = {}",
                resu.vec_id
            );
            for row in client.query(&q[..], &[]).unwrap() {
                let phrase: String = row.get("phrase");
                if !phrase_set.contains(&phrase) {
                    phrase_set.insert(phrase.clone());
                    resu.phrases.push(Phrase_Info {
                        phrase,
                        class: row.get("class"),
                    });
                }
            }
        }
    }
}
