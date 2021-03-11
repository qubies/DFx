use std::num::Wrapping;

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
