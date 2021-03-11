#![feature(test)]
#![feature(wrapping_int_impl)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// use pyo3::wrap_pyfunction;

mod dfx;
mod discretize;
mod distances;
pub mod pgres;

pub struct Vector_Search_Server {
    vs: dfx::Vec_Search,
    search_limit: usize,
    length_boost: f32,
}

impl Vector_Search_Server {
    pub fn new(num_buckets: usize, search_limit: usize, length_boost: f32, mode: String) -> Self {
        Vector_Search_Server {
            vs: dfx::Vec_Search::New(num_buckets, mode),
            search_limit,
            length_boost,
        }
    }
    fn Paraphrase(&self, embedding_in: Vec<f32>, phrase_len: usize) -> Vec<(u32, String)> {
        let mut paraphrases = Vec::new();
        for result in self.vs.Query(
            embedding_in,
            phrase_len,
            self.search_limit,
            self.length_boost,
        ) {
            for phrase in result.phrases {
                paraphrases.push((result.score, phrase.phrase));
            }
        }
        paraphrases
    }

    fn Encode(&self, vec_in: Vec<f32>) -> Vec<u64> {
        self.vs.Encode(&vec_in)
    }

    unsafe fn Compare(&self, vec_1: Vec<f32>, vec_2: Vec<f32>) -> u32 {
        let vec_1 = self.vs.Encode(&vec_1);
        let vec_2 = self.vs.Encode(&vec_2);
        dfx::Compare(&vec_1, &vec_2)
    }

    // fn KNN(&self, vec_in: Vec<u64>, k: usize) -> Vec<Vec<u64>> {
    //     self.vs.KNN(vec_in, k).iter().map(|x| x.vec).collect_vec()
    // }

    // fn Inference_Only(&self) {
    //     self.vs.Close();
    // }
}
