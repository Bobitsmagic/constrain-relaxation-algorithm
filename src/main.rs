use core::panic;
use std::env;
use datasets::SamplePoint;
use nalgebra::DVector;
use rand::{distributions, prelude::Distribution, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

mod datasets;



fn main() {
    // check_convexity();
    // create_convex_plot();

    //enable backtrace
    env::set_var("RUST_BACKTRACE", "1");

    test_basic_example();
}

fn test_basic_example() {
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    let samples = datasets::basic_example();

    let (mut labels, mut weights) = datasets::gen_parameter(&samples);
    
    let learnrate = 0.01;
    let lambda = 0.5;   

    for _ in 0..1 {
        randomize_weights(&mut weights, 1.0, &mut rng);
        randomize_lables(&mut labels, &mut rng);
    
        // println!("Loss: {}", evaluate_loss(&samples, &labels, &weights, lambda));
        // for i in 0..4 {
        //     let v = &samples[i];
        //     let x = v.dot(&weights);
        //     println!("{}: {}", i, x);
        // }
    
        gradient_descent(&samples, &mut labels, &mut weights, learnrate, lambda);
    }
}

fn randomize_lables(v: &mut DVector<f64>, rng: &mut ChaCha8Rng) {
    for i in 0..v.len() {
        v[i] = rng.gen_range(0.0..1.0);
    }
}

fn randomize_weights(v: &mut DVector<f64>, range: f64, rng: &mut ChaCha8Rng) {
    let normal = Normal::new(0.0, range).unwrap();
    for i in 0..v.len() {
        v[i] = normal.sample(rng);
    }
}

fn gradient_descent(samples: &Vec<SamplePoint>, labels: &mut DVector<f64>, weights: &mut DVector<f64>, learnrate: f64, lambda: f64) {
    let mut linear_inequalities = Vec::new();

    // println!("Start");
    // print_vector(&weights);
    // print_vector(&labels);

    let weight_count = weights.len();
    for i in 0..labels.len() {
        let mut v = DVector::zeros(weight_count + labels.len());
        v[weight_count + i] = 1.0;
        linear_inequalities.push((v.clone(), 1.0));

        v[weight_count + i] = -1.0;
        linear_inequalities.push((v, 0.0));
    }

    for _ in 0..200 {
        let (weight_grad, label_grad) = evaluate_grad(&samples, &labels, &weights, lambda);

        // println!("Weights:");
        // print_vector(&weights);
        // println!("Weight Grad:");
        // print_vector(&weight_grad);

        // println!("Label Grad:");
        // print_vector(&label_grad);

        let mut full_grad = DVector::zeros(weight_grad.len() + label_grad.len());
        for i in 0..weight_grad.len() {
            full_grad[i] = weight_grad[i];
        }

        for i in 0..label_grad.len() {
            full_grad[i + weight_grad.len()] = label_grad[i];
        }

        let mut full_pos = DVector::zeros(weight_grad.len() + label_grad.len());
        for i in 0..weight_grad.len() {
            full_pos[i] = weights[i];
        }
        for i in 0..label_grad.len() {
            full_pos[i + weight_grad.len()] = labels[i];
        }

        //Invert direction of gradient
        let step = - learnrate * full_grad;

        println!("Step:");
        print_vector(&step);

        let new_step = project_search_direction(&full_pos, &step, &linear_inequalities);

        // println!("Projection:");
        // print_vector(&new_step);

        let new_step = clip_to_constraint(&full_pos, &new_step, &linear_inequalities);

        // println!("Clip:");
        // print_vector(&new_step);

        *weights += new_step.rows(0, weight_grad.len()).clone_owned();
        *labels += new_step.rows(weight_grad.len(), label_grad.len()).clone_owned();

        println!("{}", evaluate_loss(&samples, &labels, &weights, lambda));
    }
    

    print_vector(&labels);
    // println!("Weights");
    print_vector(&weights);

    if labels.iter().any(|x| *x < 0.5) {
        // println!("Labels");

        // panic!("Labels not in range");
    }

    // println!("Weights: {:?}", weights);
}

fn print_vector(v: &DVector<f64>) {
    for x in v.iter() {
        print!("{:.3} ", x);
    }
    println!("");
}

const P: f64 = 2.0;

fn logistic_regression(r: f64, y: f64) -> f64 {
    -r*y + (1.0 + P.powf(r)).ln() / P.ln()
}

fn loss(r: f64, y: f64) -> f64 {
    // -r*y + (1.0 + P.powf(r)).ln() / P.ln()
    // -r*y*y + (1.0 + P.powf(r)).ln() / P.ln()
    16.0 * logistic_regression(r, y) + r * r
}

fn create_convex_plot() {
    const COUNT: usize = 100000000;
    let mut rng = rand::thread_rng();

    const BIN_COUNT: usize = 30;
    const GEN_RANGE: f64 = 10.0;
    const BIN_RANGE: f64 = 2.0;

    let mut bins = vec![vec![100.0_f64; BIN_COUNT]; BIN_COUNT];


    let mut next_print = 1;
    for k in 0..COUNT {
        if next_print == k {
            println!("{}%", k as f64 / COUNT as f64 * 100.0);
            next_print *= 2;
        }
        let r1 = rng.gen_range(-GEN_RANGE..GEN_RANGE);
        let y1 = rng.gen_range(0.0..1.0);
        
        let r2 = rng.gen_range(-GEN_RANGE..GEN_RANGE);
        let y2 = rng.gen_range(0.0..1.0);

        let l1 = loss(r1, y1);
        let l2 = loss(r2, y2);

        for i in 0..100 {
            let d = i as f64 / 100.0;
            let l = l1 * d + l2 * (1.0 - d);
            
            let r = r1 * d + r2 * (1.0 - d);
            let y = y1 * d + y2 * (1.0 - d);

            let r_ind = (r + BIN_RANGE) / (2.0 * BIN_RANGE) * BIN_COUNT as f64;
            let y_ind = y * BIN_COUNT as f64;

            if r_ind < 0.0 || r_ind >= BIN_COUNT as f64 || y_ind < 0.0 || y_ind >= BIN_COUNT as f64 {
                continue;
            }

            bins[r_ind as usize][y_ind as usize] = bins[r_ind as usize][y_ind as usize].min(l);
        }        
    }

    for i in 0..BIN_COUNT {
        for j in 0..BIN_COUNT {
            let r = i as f64 / BIN_COUNT as f64 * BIN_RANGE * 2.0 - BIN_RANGE;
            let y = j as f64 / BIN_COUNT as f64;
            println!("{},{},{}", r, y, bins[i][j] - loss(r, y));
        }
    }

    fn bin_index(x: f64) -> usize {
        (x / BIN_COUNT as f64) as usize
    }
}

fn check_convexity() {
    const COUNT: usize = 1000000;
    let mut rng = rand::thread_rng();

    let mut convex = 0;
    for _ in 0..COUNT {
        let r1 = rng.gen_range(-100.0..100.0);
        let y1 = rng.gen_range(0.0..1.0);
        
        let r2 = rng.gen_range(-100.0..100.0);
        let y2 = rng.gen_range(0.0..1.0);

        let l1 = loss(r1, y1);
        let l2 = loss(r2, y2);

        for i in 0..10 {
            let x = i as f64 / 10.0;      
            let r = r1 * x + r2 * (1.0 - x);
            let y = y1 * x + y2 * (1.0 - x);
            let lm = l1 * x + l2 * (1.0 - x);

            let l = loss(r, y);
            if l <= lm {
                convex += 1;
            }
        }        
    }

    println!("Convex: {}%", convex as f64 / (COUNT * 10) as f64 * 100.0);
}

fn logistic_loss(r: f64, y: f64) -> f64 {
    -r * y + (1.0 + r.exp()).ln()
}

fn logistic_loss_grad(r: f64, y: f64) -> f64 {
    1.0 / (1.0 + r.exp()) - y
}

fn evaluate_loss(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> f64 {
    let reg_loss = lambda * regularizer(weights);
    let mut sample_loss = 0.0;
    for i in 0..inputs.len() {
        let x = &inputs[i];
        let f = x.dot(&weights);

        let y = labels[i];
        sample_loss += logistic_loss(f, y);
    }

    sample_loss + reg_loss
}

fn regularizer(weights: &DVector<f64>) -> f64 {
    weights.dot(weights) * 0.5
}

fn regularizer_grad(weights: &DVector<f64>) -> DVector<f64> {
    weights.clone() 
}

fn evaluate_grad(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> (DVector<f64>, DVector<f64>) {
    let mut weight_grad = DVector::zeros(weights.len());
    let mut label_grad = DVector::zeros(inputs.len());
    
    for i in 0..inputs.len() {
        let x = &inputs[i];
        let f = x.dot(&weights);
        
        let y = labels[i];
        weight_grad += logistic_loss_grad(f, y) * x;

        label_grad[i] = -f;
    }

    // println!("## Sample Grad:");
    // print_vector(&weight_grad);

    let reg_grad = regularizer_grad(weights);
    // println!("## Regularizer Grad:");
    // print_vector(&reg_grad);

    weight_grad += lambda * reg_grad;

    (weight_grad, label_grad)
}

//step = -learnrate * gradient 
fn clip_to_constraint(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut min_alpha = 1.0;

    for (n, b) in linear_inequalities {
        let alpha = (b - n.dot(&pos)) / n.dot(&step);
        if alpha < min_alpha && step.dot(&n) > 0.0 {
            // println!("Clipping at {}", alpha);
            // // println!("Normal:");
            // print_vector(&n);

            min_alpha = alpha;
        }
    }

    step * min_alpha
}

fn project_search_direction(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut new_step = step.clone();
    
    for (n, b) in linear_inequalities {
        //if inequality is hit/violated and step is outwards
        if (n.dot(pos) - *b).abs() < 0.00001 && n.dot(&new_step) > 0.0 {
            //project step onto plane vector

            // println!("\tProjection:");
            // print!("\t");

            // print_vector(&n);

            let pos_proj = pos - (n.dot(pos) - b) * n;
            let step_proj = new_step.clone() - (n.dot(&new_step)) * n;

            //to correct overshoot of inequalities
            new_step = (pos_proj + step_proj) - pos;
            // print!("\t");
            // print_vector(&new_step);
        }   
    }

    new_step
}
