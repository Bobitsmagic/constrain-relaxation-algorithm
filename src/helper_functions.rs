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