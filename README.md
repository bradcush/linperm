# Linear\*-Time Permutation Check

## Background

A recent paper by Benedikt Bunz, Jessica Chen, and Zachary DeStefano proposing
two new permutation arguments for use in modern SNARK protocols. This
repository aims to implement `BiPerm` and `MulPerm` efficiently in Rust. It's a
work in progress which has not gone through a formal audit and is not
recommended for use in production systems. Use at your own risk.

## Paper

- [Cryptology ePrint Archive](https://eprint.iacr.org/2025/1850)
- [Linear\*-Time Permuation Check](2025-ltpc.pdf)

## Building

``` sh
cargo build
```

## Testing

Unit, integration, and doc tests:

``` sh
cargo test
```

## Organization

- `app`: Usage and integration, binary crate
- `biperm`: BiPerm implementation, library crate
- `mulperm`: MulPerm implemtation, library crate (TBD)
