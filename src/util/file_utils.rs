#![allow(dead_code)]

use std::fs::File;
use std::io::{self, Write, Read};

pub fn write_into_file(contents: &str, filepath: &str) -> io::Result<()> {
	let mut f = File::create(filepath)?;
	f.write_all(contents.as_bytes())
}

pub fn read_from_file(file_name: &str) -> io::Result<String> {
	let mut f = File::open(file_name)?;
	let mut content = String::new();
	f.read_to_string(&mut content).expect(concat!("Failed To Read string in ", file!(), line!()));
	Ok(content)
}
