use rand::prelude::*;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io;

// Gaussian distribution parameters for continuous features
#[derive(Clone, Debug)]
struct GaussianParams {
    mean: f64,
    variance: f64,
}

// Naive Bayes Classifier structure
struct NaiveBayesClassifier {
    classes: Vec<String>,
    class_priors: Vec<f64>,
    feature_params: Vec<Vec<GaussianParams>>, // [class][feature]
    is_trained: bool,
}

impl NaiveBayesClassifier {
    fn new() -> Self {
        Self {
            classes: Vec::new(),
            class_priors: Vec::new(),
            feature_params: Vec::new(),
            is_trained: false,
        }
    }

    // Train the classifier with labeled data
    fn train(&mut self, features: &[Vec<f64>], labels: &[String]) {
        if features.is_empty() || labels.is_empty() {
            return;
        }

        // Extract unique classes
        self.classes = labels.iter().cloned().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        self.classes.sort();

        let n_samples = labels.len();
        let n_features = features[0].len();
        let n_classes = self.classes.len();

        // Calculate class priors (probability of each class)
        self.class_priors = vec![0.0; n_classes];
        for label in labels {
            if let Some(idx) = self.classes.iter().position(|c| c == label) {
                self.class_priors[idx] += 1.0;
            }
        }
        for prior in &mut self.class_priors {
            *prior /= n_samples as f64;
        }

        // Calculate Gaussian parameters for each feature per class
        self.feature_params = vec![vec![GaussianParams { mean: 0.0, variance: 0.0 }; n_features]; n_classes];

        for (class_idx, class) in self.classes.iter().enumerate() {
            // Filter samples belonging to this class
            let class_samples: Vec<&Vec<f64>> = features.iter()
                .zip(labels.iter())
                .filter(|(_, l)| *l == class)
                .map(|(f, _)| f)
                .collect();

            let n_class_samples = class_samples.len();

            // Calculate mean and variance for each feature
            for feature_idx in 0..n_features {
                let values: Vec<f64> = class_samples.iter()
                    .map(|s| s[feature_idx])
                    .collect();

                let mean = values.iter().sum::<f64>() / n_class_samples as f64;
                let variance = values.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / n_class_samples as f64;

                // Add small epsilon to avoid division by zero
                let variance = variance + 1e-9;

                self.feature_params[class_idx][feature_idx] = GaussianParams { mean, variance };
            }
        }

        self.is_trained = true;
    }

    // Calculate Gaussian probability density function
    fn gaussian_pdf(&self, x: f64, mean: f64, variance: f64) -> f64 {
        let exponent = -((x - mean).powi(2)) / (2.0 * variance);
        let coefficient = 1.0 / (2.0 * std::f64::consts::PI * variance).sqrt();
        coefficient * exponent.exp()
    }

    // Predict the class for a single instance
    fn predict(&self, features: &[f64]) -> Option<(String, f64)> {
        if !self.is_trained {
            return None;
        }

        let mut max_prob = f64::NEG_INFINITY;
        let mut best_class = String::new();

        // Calculate posterior probability for each class using Bayes theorem
        for (class_idx, class) in self.classes.iter().enumerate() {
            // Start with log of prior probability
            let mut log_prob = self.class_priors[class_idx].ln();

            // Multiply by likelihood of each feature (using log for numerical stability)
            for (feature_idx, &feature_value) in features.iter().enumerate() {
                let params = &self.feature_params[class_idx][feature_idx];
                let likelihood = self.gaussian_pdf(feature_value, params.mean, params.variance);
                log_prob += likelihood.ln();
            }

            if log_prob > max_prob {
                max_prob = log_prob;
                best_class = class.clone();
            }
        }

        Some((best_class, max_prob.exp()))
    }

    // Calculate accuracy on test data
    fn accuracy(&self, features: &[Vec<f64>], labels: &[String]) -> f64 {
        let mut correct = 0;
        for (feat, label) in features.iter().zip(labels.iter()) {
            if let Some((predicted, _)) = self.predict(feat) {
                if predicted == *label {
                    correct += 1;
                }
            }
        }
        correct as f64 / labels.len() as f64
    }
}

// Generate synthetic dataset for demonstration
fn generate_dataset(n_samples: usize) -> (Vec<Vec<f64>>, Vec<String>) {
    let mut rng = thread_rng();
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        // Use sample() to avoid gen keyword
        let random_val: f64 = rng.sample(rand::distributions::Standard);
        let class = if random_val < 0.5 { "A" } else { "B" };
        
        // Generate features based on class
        let feature_vec = if class == "A" {
            vec![
                rng.sample(rand::distributions::Uniform::new(0.0, 5.0)),
                rng.sample(rand::distributions::Uniform::new(0.0, 5.0)),
            ]
        } else {
            vec![
                rng.sample(rand::distributions::Uniform::new(5.0, 10.0)),
                rng.sample(rand::distributions::Uniform::new(5.0, 10.0)),
            ]
        };

        features.push(feature_vec);
        labels.push(class.to_string());
    }

    (features, labels)
}

fn main() -> Result<(), io::Error> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut classifier = NaiveBayesClassifier::new();
    let mut logs = Vec::new();

    // Generate training data
    logs.push("🔄 Generating synthetic dataset...".to_string());
    let (train_features, train_labels) = generate_dataset(200);
    logs.push(format!("✅ Generated {} training samples", train_features.len()));

    // Train classifier
    logs.push("🧠 Training Naive Bayes classifier...".to_string());
    classifier.train(&train_features, &train_labels);
    logs.push("✅ Training completed!".to_string());

    // Generate test data
    let (test_features, test_labels) = generate_dataset(50);
    logs.push(format!("📊 Generated {} test samples", test_features.len()));

    // Calculate accuracy
    let accuracy = classifier.accuracy(&test_features, &test_labels);
    logs.push(format!("🎯 Accuracy: {:.2}%", accuracy * 100.0));

    // Make sample predictions
    logs.push("\n📋 Sample Predictions:".to_string());
    for i in 0..5.min(test_features.len()) {
        if let Some((predicted, prob)) = classifier.predict(&test_features[i]) {
            let actual = &test_labels[i];
            let correct = if predicted == *actual { "✓" } else { "✗" };
            logs.push(format!(
                "  {} Sample {}: Features=[{:.2}, {:.2}] → Predicted: {} (conf: {:.4}) | Actual: {}",
                correct, i + 1, test_features[i][0], test_features[i][1], 
                predicted, prob, actual
            ));
        }
    }

    logs.push("\n💡 Press 'q' to quit".to_string());

    // Main loop
    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(10),
                    Constraint::Length(3),
                ].as_ref())
                .split(f.size());

            // Title
            let title = Paragraph::new("🤖 Naive Bayes Classifier in Rust")
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(title, chunks[0]);

            // Logs
            let items: Vec<ListItem> = logs.iter()
                .map(|l| {
                    let style = if l.contains("✅") {
                        Style::default().fg(Color::Green)
                    } else if l.contains("🎯") {
                        Style::default().fg(Color::Yellow)
                    } else if l.contains("✓") {
                        Style::default().fg(Color::Green)
                    } else if l.contains("✗") {
                        Style::default().fg(Color::Red)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    ListItem::new(Line::from(l.as_str())).style(style)
                })
                .collect();

            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Results"));
            f.render_widget(list, chunks[1]);

            // Footer
            let footer = Paragraph::new("Controls: q = quit")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(footer, chunks[2]);
        })?;

        // Handle input
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
