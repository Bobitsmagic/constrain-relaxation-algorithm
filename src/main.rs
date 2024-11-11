use std::env;

use nalgebra::DVector;
use rand::Rng;

fn main() {
    // check_convexity();
    // create_convex_plot();

    //enable backtrace
    env::set_var("RUST_BACKTRACE", "1");

    let samples = vec![
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
        DVector::from_vec(vec![-1.0, 0.0]),
        DVector::from_vec(vec![-1.0, -1.0]),
    ];

    let mut labels = DVector::from_vec(vec![1.0, 1.0, 0.0, 0.0]);

    let mut weights = DVector::from_vec(vec![0.2, 0.1]);

    let learnrate = 0.1;
    let lambda = 0.1;

    //2 + 4 elements
    let mut linear_inequalities = Vec::new();

    for i in 0..4 {
        let mut v = DVector::zeros(6);
        v[i + 2] = 1.0;
        linear_inequalities.push((v.clone(), 1.0));

        v[i + 2] = -1.0;
        linear_inequalities.push((v, 0.0));
    }


    println!("Loss: {}", evaluate_loss(&samples, &labels, &weights));
    for i in 0..4 {
        let v = &samples[i];
        let x = v.dot(&weights);
        println!("{}: {}", i, x);
    }

    for _ in 0..10 {
        let (weight_grad, label_grad) = evaluate_grad(&samples, &labels, &weights, lambda);

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

        let step = - learnrate * full_grad;

        println!("Step:");
        print_vector(&step);

        let new_step = project_search_direction(&full_pos, &step, &linear_inequalities);

        // println!("Projection:");
        // print_vector(&new_step);

        let new_step = clip_to_constraint(&full_pos, &new_step, &linear_inequalities);

        // println!("Clip:");
        // print_vector(&new_step);

        weights += new_step.rows(0, weight_grad.len()).clone_owned();
        labels += new_step.rows(weight_grad.len(), label_grad.len()).clone_owned();
        
        println!("Loss: {}", evaluate_loss(&samples, &labels, &weights));

        
        println!("Labels");
        print_vector(&labels);
        // println!("Weights: {:?}", weights);
    }
    
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

fn logistic_loss(x: f64, y: f64) -> f64 {
    -x * y + (1.0 + x.exp()).ln()
}

fn logistic_loss_grad(x: f64, y: f64) -> (f64, f64) {
    (1.0 / (1.0 + x.exp()) - y, -x)
}

fn evaluate_loss(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>) -> f64 {
    let mut loss = 0.0;
    for i in 0..inputs.len() {
        let v = &inputs[i];
        let x = v.dot(&weights);

        let y = labels[i];
        loss += logistic_loss(x, y);
    }

    loss / inputs.len() as f64
}

fn regularizer(weights: &DVector<f64>) -> f64 {
    weights.dot(weights) * 0.5
}

fn evaluate_grad(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> (DVector<f64>, DVector<f64>) {
    let mut weight_grad = DVector::zeros(weights.len());

    //regularizer
    for i in 0..weights.len() {
        weight_grad[i] = lambda * regularizer(weights);
    }

    let mut label_grad = DVector::zeros(inputs.len());
    
    for i in 0..inputs.len() {
        let v = &inputs[i];
        let x = v.dot(&weights);
        
        let y = labels[i];
        weight_grad += logistic_loss_grad(x, y).0 * v;

        label_grad[i] = -logistic_loss(x, y);
    }

    (weight_grad / inputs.len() as f64, label_grad)
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