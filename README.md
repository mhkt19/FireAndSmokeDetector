
# Fire and Smoke Detection


![3](https://github.com/user-attachments/assets/1c40d2cd-e7cf-4ae5-9f85-b57759bcc306)


## Introduction

Fire detection using AI and computer vision has emerged as a crucial innovation for providing early warnings and mitigating the disastrous effects of fires, particularly in large industries. Traditional sensor-based fire detection systems, such as smoke detectors and heat sensors, have notable limitations in terms of range, environmental factors, and detection delays. AI-driven computer vision offers a powerful complementary solution, capable of detecting fires from visual data at their inception, even in challenging environments where traditional sensors might fail.

### Why Fire Detection with Computer Vision?

- **Early Detection**: Computer vision systems can detect fires and smoke at their very onset, providing critical early warnings.
- **Wide Area Monitoring**: Cameras can cover large areas, making it feasible to monitor expansive regions such as forests, industrial sites, and large buildings.
- **Remote Monitoring**: Vision-based systems allow for remote monitoring and control, which is vital for inaccessible or hazardous locations.
- **Integration with Existing Infrastructure**: Many places already have video surveillance systems in place, which can be leveraged for fire detection.
- **Reduction in False Alarms**: Advanced algorithms can differentiate between actual fires and other heat or smoke sources, reducing false alarms.
  
### Enhancing Safety in Large Industries

Large industries, such as manufacturing plants, refineries, and warehouses, present unique fire risks due to the presence of flammable materials, complex machinery, and expansive areas. Implementing AI-driven fire detection can transform safety protocols and emergency response strategies in these environments.

- **Comprehensive Coverage**: AI-enabled cameras provide continuous, real-time surveillance across vast industrial sites, ensuring no area is left unmonitored.

- **Rapid Response**: Early detection through AI allows for immediate response, minimizing damage and preventing potential catastrophic events.

- **Cost-Effective**: Utilizing existing video surveillance systems reduces the need for additional sensor installations, making AI fire detection a cost-effective enhancement.

- **Integration with Safety Protocols**: AI fire detection systems can be seamlessly integrated with existing safety protocols, enabling automated alerts and responses tailored to the specific needs of large industries.

### Complementing Sensor-Based Systems

While sensor-based systems are effective, their combination with computer vision can significantly enhance overall fire detection capabilities:
- **Hybrid Systems**: Integrating computer vision with traditional sensors can provide a more comprehensive fire detection system, utilizing the strengths of both methods.
- **Verification**: Vision-based systems can verify alarms from traditional sensors, ensuring that emergency responses are accurate and timely.

## Project Overview

This project implements a deep learning-based approach to detect fire and smoke in images. The dataset used for training and testing is sourced from the paper titled "[Smoke and Fire Detection Dataset](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)."

### Dataset
The dataset consists of images classified into four categories:

### Both Fire and Smoke
<p align="center">
  <img src="https://github.com/user-attachments/assets/d5fbab52-c862-4fa2-acbd-e7fac5c23148" alt="bothFireAndSmoke_CV019477" width="400"/>
  <img src="https://github.com/user-attachments/assets/8abab65f-ca2f-42b3-9eeb-0fa398bd7b35" alt="bothFireAndSmoke_CV013509" width="400"/>
</p>

### Fire



<p align="center">
  <img src="https://github.com/user-attachments/assets/e1589df7-0085-4d3c-865e-3105e04bb4a2" alt="fire_CV007057" width="400"/>
  <img src="https://github.com/user-attachments/assets/1a358272-c87a-42ce-a8eb-2a0fabaa0095" alt="fire_CV011611" width="400"/>
</p>

### Smoke

<p align="center">
  <img src="https://github.com/user-attachments/assets/ccc14ccd-be5f-46d8-b3b7-8d95056528f4" alt="smoke_CV008499" width="400"/>
  <img src="https://github.com/user-attachments/assets/41155d54-97ac-499c-a83a-f8b710cff413" alt="smoke_CV001310" width="400"/>
</p>



### Neither Fire nor Smoke


<p align="center">
  <img src="https://github.com/user-attachments/assets/2b2af6c5-a85d-4151-b4be-9f2d14ba6f94" alt="neitherFireNorSmoke_CV000893" width="400"/>
  <img src="https://github.com/user-attachments/assets/eaa4cb3b-258b-4aae-9f84-fa8d21039762" alt="neitherFireNorSmoke_CV024001" width="400"/>
</p>





Due to resource limitations, only a subset of the dataset is used for training and testing. Specifically, a `sampling_ratio = 2%` of the whole dataset is used, as specified in the `config.json` file.



<!--### Configuration

The configuration settings for the project are stored in a `config.json` file. Key parameters include:

```json
{
    "test_ratio": 0.2,  // Ratio of the dataset to be used for testing
    "sampling_ratio": 0.02,  // Ratio of data to be sampled for training and testing
    "batch_size": 32,  // Number of samples per gradient update
    "num_epochs": 15,  // Number of epochs for training
    "learning_rate": 0.001,  // Learning rate for the optimizer
    "dataset_folder": "dataset",  // Path to the dataset folder
    "train_folder": "train",  // Path to the training dataset folder
    "test_folder": "test",  // Path to the testing dataset folder
    "results_folder": "results",  // Path to the results folder
    "patience": 3,  // Patience for early stopping
    "sampling_method": "random",  // Method for sampling data (random or similarity-based)
    "max_similar_photos": 2,  // Maximum similar photos in the dataset
    "similarity_threshold": 0.8,  // Threshold for similarity-based sampling
    "random_seed": 42,  // Seed for random number generation
    "num_runs": 10,  // Number of runs for training and evaluation
    "data_augmentation": true,  // Whether to use data augmentation
    "model_choice": "resnet18",  // Model choice ('resnet18', 'resnet50', 'vgg16')
    "use_dropout": true,  // Whether to use dropout
    "use_scheduler": false  // Whether to use a learning rate scheduler
}
```-->

### Code Structure

The project is divided into several scripts:

1. **data_preparation.py**: Prepares the dataset by splitting it into training and testing sets, and performs data augmentation if enabled.
2. **model_training.py**: Defines the model architecture, training, and evaluation processes.
3. **split_photos.py**: Additional utility for splitting and organizing photos.

### Running the Project

To run the project, follow these steps:

1. **Install Dependencies**: Ensure you have Python and the required libraries installed. You can install the necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**: Place your dataset in the directory specified in the `config.json` file. Run the data preparation script to split the dataset:
   ```bash
   python data_preparation.py
   ```

3. **Train the Model**: Run the model training script to train and evaluate the model:
   ```bash
   python model_training.py
   ```

### Results

The model was trained and evaluated over 10 runs. Below are the average results from these runs:

#### Binary Classification
In this setup, the images are classified into two groups: one group consists of images containing fire, smoke, or both; the other group consists of images with neither fire nor smoke.

- **Average Binary Accuracy**: 93.63%
- **Average Binary Precision**: 93.79%
- **Average Binary Recall**: 93.63%
  

![average_test_binary_confusion_matrix](https://github.com/user-attachments/assets/be0a8887-6860-454d-afaf-1a5f7bfce897)


#### Detailed Classification
In this setup, the images are classified into four distinct categories: both fire and smoke, fire, smoke, and neither fire nor smoke.

- **Average Test Accuracy**: 87.08%
- **Average Test Precision**: 87.11%
- **Average Test Recall**: 87.08%

  

![average_test_confusion_matrix](https://github.com/user-attachments/assets/a9464de0-dddb-4cdb-861f-eeb7cee8f927)


The binary classification approach yields a higher accuracy because it simplifies the problem by combining the fire, smoke, and fire & smoke categories into a single class. However, the detailed classification provides more granular information, which can be beneficial for specific applications despite the lower accuracy.

### Conclusion

This project demonstrates the potential of computer vision in enhancing fire and smoke detection capabilities. By leveraging deep learning techniques, we can achieve high accuracy and provide a robust solution that complements traditional sensor-based systems. This approach offers significant benefits in terms of early detection, wide-area monitoring, and integration with existing surveillance infrastructure.

### References

1. "[Smoke and Fire Detection Dataset](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)"
