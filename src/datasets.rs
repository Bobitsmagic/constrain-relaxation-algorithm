use nalgebra::DVector;

pub type SamplePoint = nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>>;

pub fn gen_parameter_n_valued(samples: &Vec<SamplePoint>, n: usize) -> (DVector<f64>, DVector<f64>) {
    let lable_count = samples.len();
    let weight_count = samples[0].len();

    let labels = DVector::from_vec(vec![0.0; lable_count * n]);
    let weights = DVector::from_vec(vec![0.1; weight_count * 3]);

    (labels, weights)
}

pub fn gen_parameter_dual(samples: &Vec<SamplePoint>) -> (DVector<f64>, DVector<f64>) {
    let lable_count = samples.len();
    let weight_count = samples[0].len();

    let labels = DVector::from_vec(vec![0.0; lable_count]);
    let weights = DVector::from_vec(vec![0.1; weight_count]);

    (labels, weights)
}

pub fn basic_example() -> Vec<SamplePoint> {
    vec![
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
        DVector::from_vec(vec![-1.0, 0.0]),
        DVector::from_vec(vec![-1.0, -1.0]),
    ]
}

pub fn bias_example() -> Vec<SamplePoint> {
    vec![
        DVector::from_vec(vec![1.0, 1.0, 1.0]),
        DVector::from_vec(vec![1.0, 1.0, 2.0]),
        DVector::from_vec(vec![1.0, 2.0, 1.0]),

        DVector::from_vec(vec![1.0, 5.0, 5.0]),
        DVector::from_vec(vec![1.0, 5.0, 6.0]),
        DVector::from_vec(vec![1.0, 6.0, 5.0]),
    ]
}

pub fn trippel_example() -> Vec<SamplePoint> {
    vec![
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![-1.0, 1.0]),
        DVector::from_vec(vec![-1.0, -1.0]),
    ]
}

//Loads iris data set but only the 50 Iris-setosa and 50 Iris-versicolor to make 0-1 classification
pub fn iris_data_2() -> Vec<SamplePoint> {
    let mut lines = include_str!("../data/iris.txt").lines();

    let mut samples = Vec::new();

    for _ in 50..150 {
        let split = lines.next().unwrap().split(",").collect::<Vec<&str>>();

        let mut vals = split.iter().take(4).map(|x| x.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        vals.insert(0, 1.0); //add bias

        samples.push(DVector::from_vec(vals));
    }

    samples
}

pub fn iris_data_3() -> Vec<SamplePoint> {
    let mut lines = include_str!("../data/iris.txt").lines();

    let mut samples = Vec::new();

    for _ in 0..150 {
        let split = lines.next().unwrap().split(",").collect::<Vec<&str>>();

        let mut vals = split.iter().take(4).map(|x| x.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        vals.insert(0, 1.0); //add bias

        samples.push(DVector::from_vec(vals));
    }

    samples
}