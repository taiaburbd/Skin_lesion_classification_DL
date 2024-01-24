import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - directory_path (str): The path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def conf_matrix(conf_mat,prefix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.show()
    # Save the figure
    file_path= f'result/conf_matrix/{prefix}_conf_matrix.png'
    plt.savefig(file_path)

    return file_path

def plot_training_history(history, prefix):
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot training and validation accuracy values
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Plot training and validation loss values
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()


    file_path = f'result/plot/{prefix}_training_history.png'
    # Save the figure
    fig.savefig(file_path)

    return file_path

def log_results(prefix, start_time, end_time, test_accuracy, conf_mat, classification_rep):
    log_filename = f'log/{prefix}_log.txt'
    
    with open(log_filename, 'w') as log_file:
        log_file.write(f'Start Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}\n')
        log_file.write(f'Finish Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}\n')
        log_file.write(f'Execution Time: {end_time - start_time:.2f} seconds\n')
        log_file.write(f'Test Accuracy: {test_accuracy * 100:.2f}%\n')
        log_file.write(f'Confusion Matrix:\n{conf_mat}\n')
        log_file.write(f'Classification Report:\n{classification_rep}\n')
    return True


def evaluate_model(model, test_generator):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
    
    # Generate predictions
    predictions = model.predict(test_generator, steps=len(test_generator))
    predicted_labels = np.round(predictions)

    # Access true labels from the DataFrameIterator
    true_labels = test_generator.labels

    # Compute confusion matrix and classification report
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    classification_rep = classification_report(true_labels, predicted_labels)

    return test_loss, test_accuracy, conf_mat, classification_rep

# Save model 
def model_save(model,file_path):
    model.save(file_path)

    import os

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def save_summary_to_csv(summary_df, _file_prefix):
    date_str = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f'result/df/{_file_prefix}_summary_{date_str}.csv'
    summary_df.to_csv(csv_path, index=False)
    return csv_path
