use nalgebra::DVector;

pub fn logistic_loss(r: f64, y: f64) -> f64 {
    -r * y + (1.0 + r.exp()).ln()
}

//loss gradient with respect to r
pub fn logistic_loss_grad(r: f64, y: f64) -> f64 {
    1.0 / (1.0 + (-r).exp()) - y
}

pub fn regularizer_loss(weights: &DVector<f64>) -> f64 {
    weights.dot(weights) * 0.5
}

pub fn regularizer_loss_grad(weights: &DVector<f64>) -> DVector<f64> {
    weights.clone() 
}

pub fn evaluate_loss(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> f64 {
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

pub fn evaluate_loss_n_valued(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> f64 {
    let mut sample_loss = 0.0;

    let n = labels.len() / inputs.len();
    let weight_count = weights.len() / n;
    

    // println!("n: {}, weight_count: {}", n, weight_count);
    for i in 0..inputs.len() {
        let x = &inputs[i];

        for j in 0..n {
            let start_weight = j * weight_count;
            let end_weight = (j + 1) * weight_count;

            // println!("start_weight: {}, end_weight: {}", start_weight, end_weight);
            // println!("col_count: {}", weights.ncols());

            let weight_vec = weights.rows_range(start_weight..end_weight);

            // println!("weight_vec: {}", weight_vec);
            
            let f = x.dot(&weight_vec);
    
            let y = labels[i * n + j];
            sample_loss += logistic_loss(f, y);
        }
    }

    let reg_loss = lambda * regularizer_loss(weights);
    sample_loss + reg_loss
}

pub fn evaluate_grad(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> (DVector<f64>, DVector<f64>) {
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

pub fn evaluate_grad_n_valued(inputs: &Vec<DVector<f64>>, labels: &DVector<f64>, weights: &DVector<f64>, lambda: f64) -> (DVector<f64>, DVector<f64>) {
    let n = labels.len() / inputs.len();
    let weight_count = weights.len() / n;
    
    let mut weight_grad = DVector::zeros(weights.len());
    let mut label_grad = DVector::zeros(inputs.len() * n);
    
    for i in 0..inputs.len() {
        let x = &inputs[i];

        for j in 0..n {
            let start_weight = j * weight_count;
            let end_weight = (j + 1) * weight_count;

            let weight_vec = weights.rows_range(start_weight..end_weight);

            let f = x.dot(&weight_vec);
            
            // println!("label_count: {} index {}", labels.len(), i * n + j);
            let y = labels[i * n + j];
            let grad_update = logistic_loss_grad(f, y) * x;

            for k in start_weight..end_weight {
                weight_grad[k] += grad_update[k - start_weight];
            }
    
            label_grad[i * n + j] = -f; //loss gradient with respect to the label
        }
    }

    let reg_grad = regularizer_loss_grad(weights);

    weight_grad += lambda * reg_grad;

    (weight_grad, label_grad)
}