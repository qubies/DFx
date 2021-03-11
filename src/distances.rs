#[cfg_attr(target_arch = "x86_64", target_feature(enable = "popcnt"))]
pub unsafe fn ones(x: u64) -> u32 {
    return x.count_ones();
}

pub unsafe fn distance(a: &u64, b: &u64) -> u32 {
    return ones(a ^ b);
}

pub unsafe fn distances(a: &[u64], b: &[u64]) -> u32 {
    let mut res = 0;
    for (x, y) in a.iter().zip(b) {
        res += distance(x, y);
    }
    return res;
}
extern crate test;
#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test::Bencher;

    fn random_discete_vec(buckets: usize, dim: usize) -> Vec<u64> {
        let mut rng = rand::thread_rng();
        let cap = (buckets - 1) * dim / 64 + 1;
        let mut numbers = Vec::<u64>::with_capacity(cap);
        for _ in 0..cap {
            numbers.push(rng.gen::<u64>());
        }
        numbers
    }

    #[bench]
    fn dfx_test_compare(ben: &mut Bencher) {
        let a = random_discete_vec(2, 512);
        let b = random_discete_vec(2, 512);
        unsafe {
            ben.iter(|| distances(&a[..], &b[..]));
        }
    }

    #[test]
    fn dfx_test_distance() {
        unsafe {
            assert_eq!(distance(&18446744073709551615, &18446744073709551615), 0);
            assert_eq!(distance(&0, &18446744073709551615), 64);
        }
    }
}
