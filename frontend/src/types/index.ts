export interface AuditReportResponse {
  dataset_name: string;
  total_participants: number;
  participants: number[];
  sessions_per_participant: number;
  trials_per_session: number;
  total_trials: number;
  total_eeg_windows: number;
  total_eye_windows: number;
  eeg_feature_shape: number[];
  eye_feature_shape: number[];
  temporal_alignment_mismatches: number;
  nan_missing_values: number;
  discrete_classes_count: number;
  discrete_class_distribution: Record<string, number>;
  continuous_labels_available: boolean;
  dataset_valid: boolean;
  audit_passed: boolean;
  summary: string;
}

export interface HealthResponse {
  status: string;
  backend_version: string;
  device: string;
  model_trained: boolean;
  model_path?: string;
  dataset_ready: boolean;
  dataset_message: string;
  total_subjects: number;
  total_windows: number;
}

export interface DatasetStatusResponse {
  files_detected: number;
  files: string[];
  selected_feature_type: string;
  total_arrays: number;
  labeled_arrays: number;
  unlabeled_arrays: number;
  total_usable_windows: number;
  shape_distribution: Record<string, number>;
  class_distribution: Record<string, number>;
  validation_status: string;
  errors: string[];
  warnings: string[];
  training_permitted: boolean;
  audit_report?: AuditReportResponse;
}

export interface TrainingConfigRequest {
  feature_type: string;
  task_mode: '7_class' | '3_class' | 'regression';
  modality: 'eeg_only' | 'eeg_eye_multimodal';
  batch_size: number;
  learning_rate: number;
  epochs: number;
  dropout: number;
  optimizer: string;
  random_seed: number;
  early_stopping_patience: number;
  class_weighted_loss: boolean;
}

export interface EpochHistory {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export interface TrainingStatusResponse {
  status: 'idle' | 'running' | 'completed' | 'failed' | 'blocked';
  current_epoch: number;
  total_epochs: number;
  train_loss?: number;
  val_loss?: number;
  train_acc?: number;
  val_acc?: number;
  best_val_acc?: number;
  elapsed_seconds: number;
  error?: string;
  history: EpochHistory[];
}

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface EvaluationMetricsResponse {
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  per_class: Record<string, PerClassMetrics>;
  confusion_matrix: number[][];
  classes: string[];
  loss_history: { train: number[]; val: number[] };
  accuracy_history: { train: number[]; val: number[] };
  split_info: Record<string, any>;
  limitations_warning?: string;
}

export interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  timestamp: string;
  disclaimer: string;
}
