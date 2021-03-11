#![feature(test)]
#![feature(wrapping_int_impl)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
// use pyo3::wrap_pyfunction;

mod dfx;
mod discretize;
mod distances;
mod pgres;

#[pyclass]
pub struct Vector_Search_Server {
    vs: dfx::Vec_Search,
    search_limit: usize,
}

#[pymethods]
impl Vector_Search_Server {
    #[new]
    fn new(num_buckets: usize, search_limit: usize, mode: String) -> Self {
        Vector_Search_Server {
            vs: dfx::Vec_Search::New(num_buckets, mode),
            search_limit,
        }
    }
    fn Paraphrase(&self, _py: Python, embedding_in: Vec<f32>) -> PyResult<Vec<(u32, String)>> {
        let mut paraphrases = Vec::new();
        for result in self.vs.Query(embedding_in, self.search_limit) {
            for phrase in result.phrases {
                paraphrases.push((result.score, phrase.phrase));
            }
        }
        Ok(paraphrases)
    }

    fn Add(&mut self, phrases: Vec<String>, embeddings: Vec<Vec<f32>>, class: i32) {
        assert_eq!(phrases.len(), embeddings.len());
        for (p, e) in phrases.iter().zip(embeddings) {
            self.vs.Add(p.clone(), e, class).unwrap();
        }
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

#[pymodule]
fn DFx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vector_Search_Server>().unwrap();
    Ok(())
}
