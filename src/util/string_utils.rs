#![allow(dead_code)]

pub fn slice_to_string(slice: &[u32]) -> String {
	slice.iter().map(|highscore| highscore.to_string())
	     .collect::<Vec<String>>().join(" ")
}

pub fn line_to_slice(line: &str) -> Vec<u32> {
	line.split(" ").filter_map( |num| num.parse::<u32>().ok()).collect()
}
