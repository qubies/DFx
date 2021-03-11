use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const TO_ONES: [u64; 64] = [
    0,
    1,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
    8191,
    16383,
    32767,
    65535,
    131071,
    262143,
    524287,
    1048575,
    2097151,
    4194303,
    8388607,
    16777215,
    33554431,
    67108863,
    134217727,
    268435455,
    536870911,
    1073741823,
    2147483647,
    4294967295,
    8589934591,
    17179869183,
    34359738367,
    68719476735,
    137438953471,
    274877906943,
    549755813887,
    1099511627775,
    2199023255551,
    4398046511103,
    8796093022207,
    17592186044415,
    35184372088831,
    70368744177663,
    140737488355327,
    281474976710655,
    562949953421311,
    1125899906842623,
    2251799813685247,
    4503599627370495,
    9007199254740991,
    18014398509481983,
    36028797018963967,
    72057594037927935,
    144115188075855871,
    288230376151711743,
    576460752303423487,
    1152921504606846975,
    2305843009213693951,
    4611686018427387903,
    9223372036854775807,
];

#[derive(Clone, Copy)]
struct Bucket {
    max: f32,
    interval: f32,
    start: f32,
    end: f32,
    size: usize,
}
fn encode(token: &mut u64, num_buckets: usize, pos: usize, val: usize) {
    *token = *token | (((0 as u64) | TO_ONES[val as usize]) << ((num_buckets - 1) * pos));
}

impl Bucket {
    fn New(max: &f32, min: &f32, size: usize) -> Bucket {
        if max <= min {
            panic!("Max less than min in discretizer, no work to do.")
        }
        if size < 2 {
            panic!("Less than 2 size intervals, no discretization will take place.")
        }
        if size > 64 {
            panic!("Size too big, 64 or less.")
        }
        let interval = (max - min) / size as f32;
        let start = min + interval / 2 as f32;
        let end = max - interval / 2 as f32;
        return Bucket {
            size,
            max: *max,
            interval,
            start,
            end,
        };
    }

    fn Discretize(&self, var: f32) -> usize {
        if var > self.max {
            return self.size - 1;
        }
        let mut current = self.start;
        let mut dis: usize = 0;
        while current < var && current < self.end {
            current += self.interval;
            dis += 1;
        }
        let basement = self.start.max(current - self.interval);
        if (basement - var).abs() < (current - var).abs() {
            return dis - 1;
        }
        return dis;
    }
}

#[derive(Clone)]
pub struct Discretizer {
    size: usize,
    bucket_size: usize,
    buckets: Vec<Bucket>,
    num_flags: usize,
    num_ints: usize,
}

impl Discretizer {
    pub fn New(filename: String, bucket_size: usize) -> Discretizer {
        let maxesMins = Maxes_and_Mins_From_File(filename).unwrap();
        let maxes = &maxesMins.Maxes;
        let mins = &maxesMins.Mins;
        if maxes.len() != mins.len() {
            panic!("Maxes and mins unbalanced size.")
        }
        let b = maxes
            .iter()
            .zip(mins)
            .map(|(max, min)| Bucket::New(max, min, bucket_size))
            .collect::<Vec<Bucket>>();
        let num_flags = 64 / (bucket_size - 1);

        return Discretizer {
            size: maxes.len(),
            buckets: b,
            bucket_size,
            num_flags,
            num_ints: (maxes.len() / num_flags as usize
                + ((maxes.len() % num_flags as usize != 0) as usize * 1))
                as usize,
        };
    }

    // this is a discretizer that reads from an array rather than a file to get its max.min
    pub fn Discretizer_From_Arrays(maxes: &[f32], mins: &[f32], bucket_size: usize) -> Discretizer {
        if maxes.len() != mins.len() {
            panic!("Maxes and mins unbalanced size.")
        }
        let b = maxes
            .iter()
            .zip(mins)
            .map(|(max, min)| Bucket::New(max, min, bucket_size))
            .collect::<Vec<Bucket>>();
        let num_flags = 64 / (bucket_size - 1);

        return Discretizer {
            size: maxes.len(),
            buckets: b,
            bucket_size,
            num_flags,
            num_ints: (maxes.len() / num_flags as usize
                + ((maxes.len() % num_flags as usize != 0) as usize * 1))
                as usize,
        };
    }

    pub fn Discretize(&self, v: &[f32]) -> Vec<u64> {
        if v.len() != self.size as usize {
            panic!(
                "Asked to discretize wrong size. Got {} expected {}",
                v.len(),
                self.size
            );
        }
        let mut result = vec![0 as u64; self.num_ints as usize];
        let mut pos = 0;
        let mut int_num = 0;
        let mut bucket_number = 0;
        for x in v.iter() {
            if pos == self.num_flags {
                pos = 0;
                int_num += 1;
            }

            let num = self.buckets[bucket_number].Discretize(*x);
            bucket_number += 1;
            encode(
                &mut result[int_num as usize],
                self.bucket_size,
                pos as usize,
                num,
            );
            pos += 1;
        }
        return result;
    }
}
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct mxmn {
    pub Maxes: Vec<f32>,
    pub Mins: Vec<f32>,
}

pub fn Maxes_and_Mins_From_File<P: AsRef<Path>>(filename: P) -> Result<mxmn, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let target: mxmn = serde_json::from_reader(reader)?;
    return Ok(target);
}

extern crate test;
#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test::Bencher;

    #[test]
    fn dfx_test_discretizer() {
        let mut disc =
            Discretizer::Discretizer_From_Arrays(&vec![1.0, 1.0, 1.0], &vec![-1.0, -1.0, -1.0], 2);
        let a = vec![1.0, 0.2, 3.0];
        let b = vec![1.0, -0.9999, 0.000001];

        assert_eq!(disc.Discretize(&a)[0], 7);
        assert_eq!(disc.Discretize(&b)[0], 5);

        disc = Discretizer::Discretizer_From_Arrays(&[1.0, 1.0, 1.0], &[-1.0, -1.0, -1.0], 3);
        assert_eq!(disc.Discretize(&a)[0], 55);
        assert_eq!(disc.Discretize(&b)[0], 19);

        disc = Discretizer::Discretizer_From_Arrays(&[1.0, 1.0, 1.0], &[-1.0, -1.0, -1.0], 4);
        assert_eq!(disc.Discretize(&a)[0], 479);
        assert_eq!(disc.Discretize(&b)[0], 199);

        // check maxes
        let v = vec![1.0 as f32; 128];
        let q = vec![-1.0 as f32; 128];
        disc = Discretizer::Discretizer_From_Arrays(&v, &q, 5);
        assert_eq!(disc.Discretize(&v)[0], 18446744073709551615); // max val

        disc = Discretizer::Discretizer_From_Arrays(&[1.0, 1.0, 1.0], &[-1.0, -1.0, -1.0], 64);
        assert_eq!(disc.Discretize(&a)[0], 9223372036854775807);
        assert_eq!(disc.Discretize(&a)[2], 9223372036854775807);
        assert_eq!(disc.Discretize(&a).len(), 3);
    }

    fn random_vec(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut numbers = Vec::<f32>::with_capacity(512);
        for _ in 0..size {
            numbers.push(rng.gen::<f32>());
        }
        numbers
    }

    #[bench]
    fn dfx_single_discretizer(b: &mut Bencher) {
        let disc = Discretizer::New("mxmn.json".to_string(), 2);
        let a = random_vec(512);
        b.iter(|| disc.Discretize(&a));
    }
}
