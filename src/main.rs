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
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let samples = datasets::iris_data();

    let (mut labels, mut weights) = datasets::gen_parameter(&samples);
    
    let label_count = labels.len();
    let weight_count = weights.len();

    let learnrate = 0.01; 
    let lambda = 0.01;   //Regularization parameter

    for _ in 0..100 {
        randomize_weights(&mut weights, 1.0, &mut rng);
        randomize_lables(&mut labels, &mut rng);    

        let mut linear_inequalities = is_label_inequalities(&labels, &weights);

        linear_inequalities.push(set_label_value_zero(weights.len(), labels.len(), 0));
        linear_inequalities.push(set_label_value_one(weights.len(), labels.len(), labels.len() - 1));

        labels[0] = 0.0;
        // labels[1] = 0.0;
        // labels[2] = 0.0;

        // labels[3] = 1.0;
        // labels[4] = 1.0;
        // labels[5] = 1.0;

        // weights[0] = -2.0;
        // weights[1] = 1.0;
        // weights[2] = 1.0;

        labels[label_count- 1] = 1.0;

        // descent(&samples, &mut labels, &mut weights, learnrate, lambda, &linear_inequalities, &mut rng);
        gradient_descent(&samples, &mut labels, &mut weights, &linear_inequalities, learnrate, lambda);

        let mut zero_count = 0;
        let mut one_count = 0;

        for i in 0..label_count {
            if labels[i] < 0.5 {
                zero_count += 1;
            } else {
                one_count += 1;
            }
        }

        println!("Zero count: {} One count: {}", zero_count, one_count);

        if zero_count == 50 {
            print_vector(&labels);
        }
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

//assures that labels are in range [0, 1]
fn is_label_inequalities(labels: &DVector<f64>, weights: &DVector<f64>) -> Vec<(DVector<f64>, f64)> {
    let mut linear_inequalities = Vec::new();

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

    linear_inequalities
}


const EPSILON: f64 = 0.0001;
//assures that a label is between 0 and EPSILON
fn set_label_value_zero(weight_count: usize, label_count: usize, index: usize) -> (DVector<f64>, f64) {
    let mut v = DVector::zeros(weight_count + label_count);
    v[weight_count + index] = 1.0;
    return (v, EPSILON); //l_i <= EPSILON
}

//assures that a label is between 0 and EPSILON
fn set_label_value_one(weight_count: usize, label_count: usize, index: usize) -> (DVector<f64>, f64) {
    let mut v = DVector::zeros(weight_count + label_count);
    v[weight_count + index] = -1.0;
    return (v, EPSILON - 1.0); //l_i >= 1 - EPSILON <=> -l_i <= EPSILON - 1
}

fn descent(samples: &Vec<SamplePoint>, labels: &mut DVector<f64>, weights: &mut DVector<f64>, learnrate: f64, lambda: f64, linear_inequalities: &Vec<(DVector<f64>, f64)>, rng: &mut ChaCha8Rng) {
    let mut current_error = evaluate_loss(&samples, &labels, &weights, lambda);

    let dim_count = weights.len() + labels.len();
    let weight_count = weights.len();
    let label_count = labels.len();

    for _ in 0..2000 {
        //Create random search direction
        let mut step_direction = DVector::zeros(dim_count);
        randomize_weights(&mut step_direction, 1.0, rng);

        step_direction = step_direction.normalize();

        //Combine weights and labels to one vector
        let mut full_pos = DVector::zeros(dim_count);
        for i in 0..weight_count {
            full_pos[i] = weights[i];
        }
        for i in 0..labels.len() {
            full_pos[i + weight_count] = labels[i];
        }

        //Invert direction of gradient
        let step = -learnrate * step_direction;

        // println!("Step:");
        // print_vector(&step);

        let projected = project_search_direction(&full_pos, &step, &linear_inequalities);

        if projected.norm_squared() > step.norm_squared() {
            println!("Overshoot");
        }

        // println!("Projection:");
        // print_vector(&new_step);

        let clipped_step = clip_to_constraint(&full_pos, &projected, &linear_inequalities);

        if clipped_step.norm_squared() > projected.norm_squared() {
            println!("Clipped overshoot {} / {} {}", clipped_step.norm_squared(), projected.norm_squared(), clipped_step.norm_squared() / projected.norm_squared());
        }

        // println!("Clip:");
        // print_vector(&new_step);

        //Update weights and labels

        let new_weights = weights.clone() + clipped_step.rows(0, weight_count).clone_owned();
        let new_labels = labels.clone() + clipped_step.rows(weight_count, label_count).clone_owned();

        let new_error = evaluate_loss(&samples, &new_labels, &new_weights, lambda);

        if new_error < current_error {
            *weights = new_weights;
            *labels = new_labels;
            current_error = new_error;

            print!("Loss: {:.3}\t Lables: ", new_error);
            print_vector(&labels);
        }

    }
    
    println!("Final labels");
    print_vector(&labels);
    println!("Final weights");
    print_vector(&weights);
    println!();
}

fn gradient_descent(samples: &Vec<SamplePoint>, labels: &mut DVector<f64>, weights: &mut DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>, learnrate: f64, lambda: f64) {
    for _ in 0..2000 {
        let (weight_grad, label_grad) = evaluate_grad(&samples, &labels, &weights, lambda);

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

        // print!("Loss: {:.3}\t Lables: ", evaluate_loss(&samples, &labels, &weights, lambda));
        // print_vector(&labels);
        // print_vector(&weights);

    }
    // print!("Loss {:.3} ", evaluate_loss(&samples, &labels, &weights, lambda));
    
    // print!("Final labels ");
    // print_vector(&labels);
    // println!("Final weights");
    // print_vector(&weights);

    // println!("{:.3} x + {:.3} y <= {:.3}", weights[1], weights[2], -weights[0]);
    // println!();
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

        // print!("## Sample:");
        // print_vector(&x);

        let f = x.dot(&weights);

        let y = labels[i];
        weight_grad += logistic_loss_grad(f, y) * x;

        label_grad[i] = -f; //loss gradient with respect to the label
    }

    // print!("## Weight grad sum:");
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
        if alpha < min_alpha && step.dot(&normal) > 0.001 && alpha >= 0.0 {
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
