use core::panic;
use std::{env, process};
use datasets::SamplePoint;
use helper_functions::{evaluate_grad, evaluate_grad_n_valued, evaluate_loss, evaluate_loss_n_valued, logistic_loss, logistic_loss_grad, regularizer_loss, regularizer_loss_grad};
use nalgebra::{one, DVector};
use rand::{prelude::Distribution, seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

mod datasets;
// mod lp_solver;
mod helper_functions;
mod alternating;


fn main() {
    //enable backtrace
    env::set_var("RUST_BACKTRACE", "1");

    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let mut samples = datasets::load_mnist_data();

    println!("Loaded {} samples {} dims", samples.len(), samples[0].len());

    
    let mut losses = Vec::new();
    let mut best_loss = f64::MAX;

    let start_time = std::time::Instant::now();
    for i in 0..1_u32 {
        samples.shuffle(&mut rng);
        let loss = alternating::solve_alternating(&samples, 3, samples.len() / 10);

        losses.push(loss);
        if loss < best_loss {
            best_loss = loss;
            println!("New best loss: {:.5}", best_loss);
        }

        if i.count_ones() == 1 {
            println!("Iteration {}: Loss: {:.5}", i, loss);
        }
    }
    println!("Time taken: {:.2?}", start_time.elapsed());

    let mut min = f64::MAX;
    let mut max = f64::MIN;

    for &loss in &losses {
        if loss < min {
            min = loss;
        }
        if loss > max {
            max = loss;
        }
    }

    let mut bins = vec![0; 20];
    for loss in losses {
        let index = ((loss - min) / (max - min) * 20.0) as usize;
        if index < bins.len() {
            bins[index] += 1;
        }
    }

    println!("Losses: min: {:.2}, max: {:.2}", min, max);

    println!("Loss distribution:");
    for (i, &count) in bins.iter().enumerate() {
        let range_start = min + (max - min) * i as f64 / 20.0;
        let range_end = min + (max - min) * (i + 1) as f64 / 20.0;
        println!("{:.2} - {:.2}: {}", range_start, range_end, count);
    }

    // test_2_valued_example();
    // test_n_valued_example();
}