#%%
# ==================== TEST ====================
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from pathlib import Path
import seaborn as sns

TRAIN_DIR = 'files/FaceMaskDataset/train'
TEST_DIR = 'files/FaceMaskDataset/test'
# MODELS_DIR = 'files/trained_models'
# LOGS_DIR = 'files/logs'
# if 'google.colab' in sys.modules:
#     TRAIN_DIR = 'FaceMaskDataset/train'
#     TEST_DIR = 'FaceMaskDataset/test'
PATH = Path('files/check2')
EPOCHS = 50
NAME = f'bigger_{EPOCHS}'
MODELS_DIR = PATH / 'trained_models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PATH / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 16

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import gc

# ============================================
# 1. LOAD THE BEST MODEL
# ============================================

def load_best_model(model_path=MODELS_DIR / 'best.keras'):
    """
    Loads the best saved model from checkpoint.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return None

    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    print(f"   ============== {model_path.name} ==============")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total parameters: {model.count_params():,}")

    return model

# ============================================
# 2. CREATE TEST GENERATOR
# ============================================

def create_test_generator(test_dir, target_size=(224, 224), batch_size=32):
    """
    Creates a test generator WITHOUT augmentation.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # IMPORTANT: Don't shuffle test set
    )

    print(f"Class indices: {test_generator.class_indices}")
    print(f"\n📊 Test Set Information:")
    print(f"   Total images: {test_generator.samples}")
    print(f"   Classes: {test_generator.class_indices}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(test_generator)}")

    return test_generator

# ============================================
# 3. EVALUATE THE MODEL
# ============================================

def evaluate_model(model, test_generator):
    """
    Evaluates the model on the test set and returns predictions.
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)

    # Reset generator to start from beginning
    test_generator.reset()

    # Get predictions and labels
    predictions = model.predict(test_generator, verbose=1)

    # Convert probabilities to binary labels (threshold 0.5)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Get true labels
    true_classes = test_generator.classes

    # Get class names
    class_names = list(test_generator.class_indices.keys())

    # Calculate metrics
    test_generator.reset()
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    f1 = f1_score(true_classes, predicted_classes)

    print(f"\n📊 Test Results:")
    print(f"   Test Loss: {loss:.4f}")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score: {f1:.4f}")

    # Detailed classification report
    print("\n📋 Classification Report:")
    print("-" * 60)
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    ))

    return {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'loss': loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'class_names': class_names
    }

# ============================================
# 4. CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(true_classes, predicted_classes, class_names):
    """
    Plots confusion matrix.
    """
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    #plt.savefig('files/confusion_matrix.png', dpi=150)
    plt.show()
    #print("✅ Confusion matrix saved to: confusion_matrix.png")

    return cm

# ============================================
# 5. MISCLASSIFICATION ANALYSIS
# ============================================

def analyze_misclassifications(test_generator, true_classes, predicted_classes,
                              class_names, num_examples=5):
    """
    Shows examples of misclassified images.
    """
    # Find misclassified indices
    misclassified_idx = np.where(true_classes != predicted_classes)[0]

    if len(misclassified_idx) == 0:
        print("\n🎉 Perfect! No misclassifications on the test set!")
        return

    print(f"\n🔍 Misclassification Analysis:")
    print(f"   Total misclassified: {len(misclassified_idx)} / {len(true_classes)}")
    print(f"   Misclassification rate: {len(misclassified_idx)/len(true_classes)*100:.2f}%")

    # Get some misclassified examples
    num_examples = min(num_examples, len(misclassified_idx))
    sample_indices = np.random.choice(misclassified_idx, num_examples, replace=False)

    # Reset generator and get batch info
    test_generator.reset()
    images = []
    labels = []

    for i in range(len(test_generator)):
        batch_images, batch_labels = next(test_generator)
        images.extend(batch_images)
        labels.extend(batch_labels)

    images = np.array(images)
    labels = np.array(labels)

    # Display misclassified examples
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 4))
    if num_examples == 1:
        axes = [axes]

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        ax.imshow(images[idx])

        true_label = class_names[true_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]

        ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                     color='red', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('files/misclassifications.png', dpi=150)
    plt.show()
    print("✅ Misclassification examples saved to: misclassifications.png")


# ============================================
# 7. MAIN EXECUTION
# ============================================

def load_all_models(models_dir=MODELS_DIR):
    
    models_dir = Path(models_dir)
    model_files = list(models_dir.glob('*.keras'))
    
    if not model_files:
        return {}
    
    models = {}
    for model_path in model_files:
        try:
            print(f"\n✅ Loading: {model_path.name}")
            model = load_best_model(model_path)
            models[model_path.stem] = model
        except Exception as e:
            print(f"Failed to load {model_path.name}: {e}")
    
    return models

def main():
    

    # 2. Create test generator
    test_generator = create_test_generator(TEST_DIR, (224, 224), BATCH_SIZE)

    models_dir = Path(MODELS_DIR)
    model_files = list(models_dir.glob('*.keras'))

    for model_path in model_files:

        try:
            model = load_best_model(model_path)
            #models[model_path.stem] = model
        except Exception as e:
            print(f"Failed to load {model_path.name}: {e}")

        # 3. Evaluate
        results = evaluate_model(model, test_generator)

        # 4. Confusion matrix
        plot_confusion_matrix(
            results['true_classes'],
            results['predicted_classes'],
            results['class_names']
        )

        del model
        del results
        gc.collect()  # Force garbage collection
        tf.keras.backend.clear_session()  # Clear TF session
        # 5. Misclassification analysis
        # analyze_misclassifications(
        #     test_generator,
        #     results['true_classes'],
        #     results['predicted_classes'],
        #     results['class_names']
        # )

        # 6. Summary
        # print("\n" + "="*60)
        # print("📊 FINAL SUMMARY")
        # print("="*60)
        # print(f"✓ Best model: files/trained_models/best.keras")
        # print(f"✓ Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        # print(f"✓ F1 Score: {results['f1_score']:.4f}")
        # print(f"✓ Test Loss: {results['loss']:.4f}")
        # print(f"✓ Total test samples: {len(results['true_classes'])}")
        # print(f"✓ Misclassified: {np.sum(results['true_classes'] != results['predicted_classes'])}")
        # print("="*60)

#return model, results

if __name__ == "__main__":
    # Run full evaluation
    #model, results = main()
    main()

    # Optional: Test on individual images
    # if model:
    #     test_single_images(model, TEST_DIR)
# %%
