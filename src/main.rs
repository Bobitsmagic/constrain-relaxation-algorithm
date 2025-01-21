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
    let mut rng = ChaCha8Rng::seed_from_u64(3);

    let samples = datasets::iris_data();

    let (mut labels, mut weights) = datasets::gen_parameter(&samples);
    
    let label_count = labels.len();

    let learnrate = 0.01; 
    let lambda = 0.01;   //Regularization parameter

    let it_count = 1000;
    let mut correct = 0;

    for _ in 0..it_count {
        randomize_weights(&mut weights, 1.0, &mut rng);
        randomize_lables(&mut labels, &mut rng);    

        let linear_inequalities = is_label_inequalities(label_count);

        let mut linear_equalities = Vec::new();

        linear_equalities.push(set_label_value_zero(label_count, 0));
        linear_equalities.push(set_label_value_one(label_count, label_count - 1));

        labels[0] = 0.0;
        labels[label_count- 1] = 1.0;

        // descent(&samples, &mut labels, &mut weights, learnrate, lambda, &linear_inequalities, &mut rng);
        gradient_descent(&samples, &mut labels, &mut weights, &linear_inequalities, &linear_equalities, learnrate, lambda);

        let mut correct_count = 0;

        for i in 0..label_count {
            if labels[i] < 0.5 {
                correct_count += (i < 50) as i32;
            } else {
                correct_count += (i >= 50) as i32;
            }
        }

        if correct_count == 100 {
            correct += 1;
        }

        // println!("Correct count: {}", correct_count);

        // if correct_count >= 95 {
        //     print_vector(&labels);
        // }
    }

    println!("Correct: {}/{} = {:.2}%", correct, it_count, correct as f64 / it_count as f64 * 100.0);
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

//assures that labels are in range [0, 1]
fn is_label_inequalities(label_count: usize) -> Vec<(DVector<f64>, f64)> {
    let mut linear_inequalities = Vec::new();

    //Add constraints for labels to be in range [0, 1]
    //They have the form: w_1 * x_1 + ... + w_N * x_N + w_(N + 1) * l_1 + ... + w_(N + L) <= Bias

    for i in 0..label_count {
        let mut v = DVector::zeros(label_count);
        v[i] = 1.0;
        linear_inequalities.push((v.clone(), 1.0)); //l_i <= 1

        v[i] = -1.0;
        linear_inequalities.push((v, 0.0)); //-l_i <= 0
    }

    linear_inequalities
}

//assures that a label is between 0 and EPSILON
fn set_label_value_zero(label_count: usize, index: usize) -> (DVector<f64>, f64) {
    let mut v = DVector::zeros(label_count);
    v[index] = 1.0;
    return (v, 0.0); //l_i == 0
}

//assures that a label is between 0 and EPSILON
fn set_label_value_one(label_count: usize, index: usize) -> (DVector<f64>, f64) {
    let mut v = DVector::zeros(label_count);
    v[index] = 1.0;
    return (v, 1.0); //l_i == 1
}

fn gradient_descent(samples: &Vec<SamplePoint>, labels_pos: &mut DVector<f64>, weights: &mut DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>, linear_equalities: &Vec<(DVector<f64>, f64)>, learnrate: f64, lambda: f64) {
    for _ in 0..3000 {
        let (weight_grad, label_grad) = evaluate_grad(&samples, &labels_pos, &weights, lambda);

        //Invert direction of gradient
        let label_step = -learnrate * label_grad;
        let eq_projected_step = project_search_direction_eq(&labels_pos, &label_step, &linear_equalities);
        let projected_step = project_search_direction_ineq(&labels_pos, &eq_projected_step, &linear_inequalities);
        let clipped_step = clip_to_constraint(&labels_pos, &projected_step, &linear_inequalities);

        //Update weights and labels
        *weights -= learnrate * weight_grad;
        *labels_pos += clipped_step;
    }
}

fn logistic_loss(r: f64, y: f64) -> f64 {
    -r * y + (1.0 + r.exp()).ln()
}

//loss gradient with respect to r
fn logistic_loss_grad(r: f64, y: f64) -> f64 {
    1.0 / (1.0 + (-r).exp()) - y
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

    let reg_grad = regularizer_loss_grad(weights);

    weight_grad += lambda * reg_grad;

    (weight_grad, label_grad)
}

//Walk as far as possible until a constraint is hit
fn clip_to_constraint(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut min_alpha = 1.0;

    //Find first contraint that is hit
    for (normal, bias) in linear_inequalities {
        let alpha = (bias - normal.dot(&pos)) / normal.dot(&step);
        if alpha < min_alpha && step.dot(&normal) > 0.001 && alpha >= 0.0 {
            min_alpha = alpha;
        }
    }

    step * min_alpha

}

fn project_search_direction_ineq(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
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

fn project_search_direction_eq(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut new_step = step.clone();
    
    //project step onto plane of inequalities
    for (normal, bias) in linear_inequalities {        
        //project pos onto plane if inequality is violated
        let pos_proj = pos - (normal.dot(pos) - bias) * normal;
        //project step onto plane
        let step_proj = new_step.clone() - (normal.dot(&new_step)) * normal;

        //to correct overshoot of inequalities
        new_step = (pos_proj + step_proj) - pos;
    }

    new_step
}

fn print_vector(v: &DVector<f64>) {
    for x in v.iter() {
        print!("{:.3} ", x);
    }
    println!("");
}
