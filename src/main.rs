use std::env;
use datasets::SamplePoint;
use nalgebra::DVector;
use rand::{prelude::Distribution, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

mod datasets;



fn main() {
    //enable backtrace
    env::set_var("RUST_BACKTRACE", "1");
    
    test_basic_example();
}

fn test_basic_example() {
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    let samples = datasets::basic_example();

    let (mut labels, mut weights) = datasets::gen_parameter(&samples);
    
    let learnrate = 0.01; 
    let lambda = 0.5;   //Regularization parameter

    for _ in 0..1 {
        randomize_weights(&mut weights, 1.0, &mut rng);
        randomize_lables(&mut labels, &mut rng);    

        gradient_descent(&samples, &mut labels, &mut weights, learnrate, lambda);
    }
}

//Randomizes labels to be uniformly distributed in [0, 1]
fn randomize_lables(v: &mut DVector<f64>, rng: &mut ChaCha8Rng) {
    for i in 0..v.len() {
        v[i] = rng.gen_range(0.0..1.0);
    }
}

//Randomizes weights to be normally distributed
fn randomize_weights(v: &mut DVector<f64>, deviation: f64, rng: &mut ChaCha8Rng) {
    let normal = Normal::new(0.0, deviation).unwrap();
    for i in 0..v.len() {
        v[i] = normal.sample(rng);
    }
}

fn gradient_descent(samples: &Vec<SamplePoint>, labels: &mut DVector<f64>, weights: &mut DVector<f64>, learnrate: f64, lambda: f64) {
    let mut linear_inequalities = Vec::new();

    // println!("Start weights");
    // print_vector(&weights);
    // print_vector(&labels);

    //Add constraints for labels to be in range [0, 1]
    //They have the form: w_1 * x_1 + ... + w_N * x_N + w_(N + 1) * l_1 + ... + w_(N + L) <= Bias

    let weight_count = weights.len();
    for i in 0..labels.len() {
        let mut v = DVector::zeros(weight_count + labels.len());
        v[weight_count + i] = 1.0;
        linear_inequalities.push((v.clone(), 1.0)); //l_i <= 1

        v[weight_count + i] = -1.0;
        linear_inequalities.push((v, 0.0)); //-l_i <= 0
    }

    for _ in 0..200 {
        let (weight_grad, label_grad) = evaluate_grad(&samples, &labels, &weights, lambda);

        // println!("Weights:");
        // print_vector(&weights);
        // println!("Weight Grad:");
        // print_vector(&weight_grad);

        // println!("Label Grad:");
        // print_vector(&label_grad);

        //Combine weight and label gradients to one vector
        let mut full_grad = DVector::zeros(weight_grad.len() + label_grad.len());
        for i in 0..weight_grad.len() {
            full_grad[i] = weight_grad[i];
        }

        for i in 0..label_grad.len() {
            full_grad[i + weight_grad.len()] = label_grad[i];
        }

        //Combine weights and labels to one vector
        let mut full_pos = DVector::zeros(weight_grad.len() + label_grad.len());
        for i in 0..weight_grad.len() {
            full_pos[i] = weights[i];
        }
        for i in 0..label_grad.len() {
            full_pos[i + weight_grad.len()] = labels[i];
        }

        //Invert direction of gradient
        let step = -learnrate * full_grad;

        // println!("Step:");
        // print_vector(&step);

        let new_step = project_search_direction(&full_pos, &step, &linear_inequalities);

        // println!("Projection:");
        // print_vector(&new_step);

        let new_step = clip_to_constraint(&full_pos, &new_step, &linear_inequalities);

        // println!("Clip:");
        // print_vector(&new_step);

        //Update weights and labels
        *weights += new_step.rows(0, weight_grad.len()).clone_owned();
        *labels += new_step.rows(weight_grad.len(), label_grad.len()).clone_owned();

        print!("Loss: {:.3}\t Lables: ", evaluate_loss(&samples, &labels, &weights, lambda));
        print_vector(&labels);
    }
    
    println!("Final labels");
    print_vector(&labels);
    println!("Final weights");
    print_vector(&weights);
    println!();
}

fn print_vector(v: &DVector<f64>) {
    for x in v.iter() {
        print!("{:.3} ", x);
    }
    println!("");
}

fn logistic_loss(r: f64, y: f64) -> f64 {
    -r * y + (1.0 + r.exp()).ln()
}

//loss gradient with respect to r
fn logistic_loss_grad(r: f64, y: f64) -> f64 {
    1.0 / (1.0 + r.exp()) - y
}

fn regularizer_loss(weights: &DVector<f64>) -> f64 {
    weights.dot(weights) * 0.5
}

fn regularizer_loss_grad(weights: &DVector<f64>) -> DVector<f64> {
    weights.clone() 
}

fn evaluate_loss(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> f64 {
    let mut sample_loss = 0.0;
    for i in 0..inputs.len() {
        let x = &inputs[i];
        let f = x.dot(&weights);
        
        let y = labels[i];
        sample_loss += logistic_loss(f, y);
    }
    
    let reg_loss = lambda * regularizer_loss(weights);
    sample_loss + reg_loss
}

fn evaluate_grad(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> (DVector<f64>, DVector<f64>) {
    let mut weight_grad = DVector::zeros(weights.len());
    let mut label_grad = DVector::zeros(inputs.len());
    
    for i in 0..inputs.len() {
        let x = &inputs[i];
        let f = x.dot(&weights);
        
        let y = labels[i];
        weight_grad += logistic_loss_grad(f, y) * x;

        label_grad[i] = -f; //loss gradient with respect to the label
    }

    // println!("## Sample Grad:");
    // print_vector(&weight_grad);

    let reg_grad = regularizer_loss_grad(weights);
    // println!("## Regularizer Grad:");
    // print_vector(&reg_grad);

    weight_grad += lambda * reg_grad;

    (weight_grad, label_grad)
}

//Walk as far as possible until a constraint is hit
fn clip_to_constraint(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut min_alpha = 1.0;

    //Find first contraint that is hit
    for (normal, bias) in linear_inequalities {
        let alpha = (bias - normal.dot(&pos)) / normal.dot(&step);
        if alpha < min_alpha && step.dot(&normal) > 0.0 {
            min_alpha = alpha;
        }
    }

    step * min_alpha
}

fn project_search_direction(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut new_step = step.clone();
    
    //project step onto plane of inequalities
    for (normal, bias) in linear_inequalities {
        //pos is on inequality and inequality is hit or violated 
        if (normal.dot(pos) - *bias).abs() < 0.00001 && normal.dot(&new_step) > 0.0 {
            //project pos onto plane if inequality is violated
            let pos_proj = pos - (normal.dot(pos) - bias) * normal;
            //project step onto plane
            let step_proj = new_step.clone() - (normal.dot(&new_step)) * normal;

            //to correct overshoot of inequalities
            new_step = (pos_proj + step_proj) - pos;
        }   
    }

    new_step
}
