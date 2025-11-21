import numpy as np
import math
import os
import scipy.io
import glob
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

import pickle
from toolbox.functions.Functions import load_npz, extract_window_size, calc_mean_std



def plot_wavelet_coefficients(coefficients, wavelet_name):
    cA3, cD3, cD2, cD1 = coefficients

    levels = [f'cA{i}' for i in range(1, len(cA3))]  

    for i, (coeff, level) in enumerate(zip(coefficients, levels), start=1):
        
        plt.plot(coeff, color='red')
        plt.title(f'{level} Coefficients')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

def plot_history(path, history, fold):
    # Loss progress during training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure('Loss')
    plt.plot(epochs, loss, 'c', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/'+'Loss_CV'+str(fold+1)+'.jpg')
    plt.close()
    plt.show()

    # Accuracy progress during training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    plt.figure('Accuracy')
    plt.plot(epochs, acc, 'c', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    #plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path+'/'+'Accuracy_CV'+str(fold+1)+'.jpg')
    plt.close()
    #plt.show()
    
    
def plot_results(path, y_true,y_pred,fold,train_text,time_taken):
    # Accuracy and F1-Score for the test set
    f1_score=metrics.f1_score(y_true, y_pred, average='macro')
    f1_score_w=metrics.f1_score(y_true, y_pred, average='weighted')
    accuracy=metrics.accuracy_score(y_true, y_pred)
    print ("Accuracy = ", accuracy)
    print ("F1_score = ", f1_score)
    print ("F1_score weighted= ", f1_score_w)
    
    if (time_taken!=0):
        f = open(path+'accuracy_f1score_CV'+str(fold+1)+'.txt', 'w')
    else:
        f = open(path+'accuracy_f1score_CV'+str(fold+1)+'.txt', 'a')
        
    f.write(train_text+'Accuracy = '+str(accuracy)+'\n')
    f.write(train_text+'F1_score = '+str(f1_score)+'\n')
    f.write(train_text+'F1_score weighted = '+str(f1_score_w)+'\n')
    
    if (time_taken!=0):
        f.write('Time taken = '+str(time_taken)+'s\n')
        f.write('Time taken/sample = '+str(1000*time_taken/len(y_pred))+'ms\n')
    f.close()
    
    # Confusion matrix for the test set
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(train_text+'Confusion Matrix')
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")
    plt.savefig(path+train_text+'Confusion_Matrix_CV'+str(fold+1)+'.jpg')
    plt.close()
    
# Function to create a boxplot for accuracy
import matplotlib.pyplot as plt
import numpy as np

def create_accuracy_boxplot(accuracy_data, window_sizes, animale, data_directory,  T_or_V, ymin ):
    # Create the boxplot for Accuracy
    width = 0.1  # Adjust the width of the boxes
    spacing = 0.2  # Adjust the distance between boxes

    plt.figure(figsize=(10, 6))
    
    # Adjust positions for each box
    positions = [i * (width + spacing) + 1 for i in range(len(window_sizes))]
    plt.boxplot(accuracy_data, positions=positions, widths=width)
    
    plt.xticks(positions, window_sizes, fontsize=15)
    plt.yticks(fontsize=16)
    plt.xlabel('Window Size (ms)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    title = f'{T_or_V} Accuracy Comparison for {animale}'
    #plt.title(title, fontsize=20)
    plt.ylim(ymin, 100)
    plt.grid(True)
    # Lists to store mean and std values for each boxplot
    mean_values = []
    std_values = []
    # Annotate mean and standard deviation on the boxplots
    for i, acc_data in enumerate(accuracy_data):
        mean_accuracy = np.mean(acc_data)
        std_accuracy = np.std(acc_data, ddof=1)
        plt.text(positions[i] - 0.12, mean_accuracy -2, f'{mean_accuracy:.1f}\n± {std_accuracy:.1f}', fontsize=15, ha='center', va='center')
        # Append mean and std values to lists
        mean_values.append(mean_accuracy)
        std_values.append(std_accuracy)
        x_value = plt.xticks()[0][i] + 0.5  # Get x-tick position
        
           
    plt.fill_between(positions, np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), color='#ff7f0e', alpha=0.1)
    
    plt.savefig(f"{data_directory}/{title}.png")
    plt.show()

# Example usage:
# create_accuracy_boxplot(accuracy_data, window_sizes, 'Animal 2', 'data_directory', 'Training', 80)

# Function to create a boxplot for F1-score
#colorAnimal = ['#1f77b4', '#ff7f0e', '#2ca02c',  '#d62728']


def create_f1score_boxplot(f1score_data, window_sizes, animale, data_directory, T_or_V, ymin):
    # Create the boxplot for F1-score
    width = 0.1  # Adjust the width of the boxes
    spacing = 0.2  # Adjust the distance between boxes

    plt.figure(figsize=(10, 6))
    
    # Adjust positions for each box
    positions = [i * (width + spacing) + 1 for i in range(len(window_sizes))]
    
    plt.boxplot(f1score_data, positions=positions, widths=width)

    plt.xticks(positions, window_sizes, fontsize=15)
    plt.yticks(fontsize=16)
    plt.xlabel('Window Size (ms)', fontsize=16)
    plt.ylabel('F1-score (%)', fontsize=16)
    title = f'{T_or_V} F1-score Comparison for {animale}'
    # plt.title(title, fontsize=20)
    plt.ylim(ymin, 100)
    plt.grid(True)

    # Lists to store mean and std values for each boxplot
    mean_values = []
    std_values = []

    # Annotate mean and standard deviation on the boxplots
    for i, f1s_data in enumerate(f1score_data):
        mean_f1score = np.mean(f1s_data)
        std_f1score = np.std(f1s_data, ddof=1)
        
        # Slightly move the plotted text to the left
        plt.text(positions[i] - 0.12, mean_f1score - 2, f'{mean_f1score:.1f}\n± {std_f1score:.1f}', fontsize=15, ha='center', va='center')
        
        # Append mean and std values to lists
        mean_values.append(mean_f1score)
        std_values.append(std_f1score)

    plt.fill_between(positions, np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), color='#ff7f0e', alpha=0.1)

    plt.savefig(f"{data_directory}/{title}.png")
    plt.show()


def get_animal_data(path_classifier, T_or_V):
    
    # Dictionary to store the data for each animal
    animal_data = {}
    
    # Loop through the animal folders and load the data
    for animale_folder in ['Animal 1', 'Animal 2', 'Animal 3']:
        
        data_directory=path_classifier+'/'+ animale_folder
        os.makedirs(data_directory, exist_ok=True)

        # Get a list of all .npz files in the directory
        file_list = os.listdir(data_directory)
        npz_files = [file_name for file_name in file_list if file_name.endswith('.npz')]
        
        # Sort npz_files in descending order based on the window size
        npz_files.sort(key=extract_window_size, reverse=True)
        
        npz_files = [file for file in npz_files if T_or_V in file]
        # Lists to store window sizes, accuracy, and f1score data
        window_sizes = []
        accuracy_data = []
        f1score_data = []

        # Load data from each .npz file and collect the window sizes, accuracy, and f1score
        for file_name in npz_files:
            accuracy, f1score = load_npz(os.path.join(data_directory, file_name))
            window_sizes.append(extract_window_size(file_name))
            accuracy_data.append(accuracy * 100)
            f1score_data.append(f1score * 100)

        # Store the data in the animal_data dictionary
        animal_data[animale_folder] = {
            'window_sizes': window_sizes,
            'accuracy_data': accuracy_data,
            'f1score_data': f1score_data
        }

    return animal_data



def w_compare_acc_boxplots(accuracy_w500, accuracy_w200, accuracy_w100, accuracy_w50, path_boxplots, T_or_V, ymin):
    ymin=ymin


    # Definiamo le posizioni dei boxplot
    positions_old = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    
    offset = 0
    positions = [pos + offset for pos in positions_old]

    # Creazione delle etichette che si ripetono
    etichette = ['An. 1',  'An. 2', 'An. 3', 'An. 1',  'An. 2', 'An. 3', 'An. 1',  'An. 2', 'An. 3', 'An. 1',  'An. 2', 'An. 3' ]

    # Definiamo la larghezza dei boxplot
    width = 0.3

    plt.figure(figsize=(15, 8))

    # Colori dei boxplot
    box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c',  '#d62728']

    # Boxplot per 500ms
    boxplot_500ms = plt.boxplot(accuracy_w500, positions=positions[:3], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[0]))

    # Boxplot per 200ms
    boxplot_200ms = plt.boxplot(accuracy_w200, positions=positions[3:6], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[1]))

    # Boxplot per 100ms
    boxplot_100ms = plt.boxplot(accuracy_w100, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))

    # Boxplot per 50ms
    boxplot_50ms = plt.boxplot(accuracy_w50, positions=positions[9:], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[3]))

    # Assegniamo i colori ai boxplot in base ai gruppi di dati
    for boxplot, color in zip([boxplot_500ms, boxplot_200ms, boxplot_100ms, boxplot_50ms], box_colors):
        for box in boxplot['boxes']:
            box.set(facecolor=color)

    plt.ylabel('Accuracy (%)', fontsize=18)
    title=f'{T_or_V} Boxplot of Accuracies, windows comparison'
    #plt.title(title, fontsize=16)
    plt.ylim(ymin, 100)
    plt.yticks(fontsize=18)
    plt.xticks([])
    plt.grid(True)

    # Annotazione media e deviazione standard per ciascun boxplot
    all_accuracy_data = accuracy_w500 + accuracy_w200 + accuracy_w100 + accuracy_w50
    for i, accuracy_data in enumerate(all_accuracy_data):
        mean = np.mean(accuracy_data)
        std = np.std(accuracy_data, ddof=1)
        plt.text(positions[i]-0.4, mean-2, f'{mean:.1f}\n±{std:.1f}', ha='center', va='center', fontsize=15)
        plt.text(positions[i], ymin-2, etichette[i], ha='center', va='center', fontsize=15)  # Aggiungo l'etichetta

    plt.text(2, ymin-4, '500ms', ha='center', va='center', fontsize=18)
    plt.text(6, ymin-4, '200ms', ha='center', va='center', fontsize=18)
    plt.text(10, ymin-4, '100ms', ha='center', va='center', fontsize=18)
    plt.text(14, ymin-4, '50ms', ha='center', va='center', fontsize=18)
    
        # Fill between for each window size
    for i, accuracy_data in enumerate([accuracy_w500, accuracy_w200, accuracy_w100, accuracy_w50]):
        mean_values = [np.mean(data) for data in accuracy_data]
        std_values = [np.std(data, ddof=1) for data in accuracy_data]
        
        plt.fill_between(positions[i * 3:(i + 1) * 3], np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), color=box_colors[i], alpha=0.1)


    plt.savefig(f"{path_boxplots}/{title}.png")
    plt.show()



def w_compare_f1_boxplots(f1score_w500, f1score_w200, f1score_w100, f1score_w50, path_boxplots, T_or_V, ymin):

    ymin=ymin
    # Definiamo le posizioni dei boxplot
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]

    # Creazione delle etichette che si ripetono
        #etichette = ['Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3' ]
    etichette = ['An. 1',  'An. 2', 'An. 3', 'An. 1',  'An. 2', 'An. 3', 'An. 1',  'An. 2', 'An. 3', 'An. 1',  'An. 2', 'An. 3' ]

    # Definiamo la larghezza dei boxplot
    width = 0.3

    plt.figure(figsize=(15, 8))

    # Colori dei boxplot
    box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c',  '#d62728']
    
    # Boxplot per 500ms
    boxplot_500ms = plt.boxplot(f1score_w500, positions=positions[:3], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[0]))

    # Boxplot per 200ms
    boxplot_200ms = plt.boxplot(f1score_w200, positions=positions[3:6], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[1]))

    # Boxplot per 100ms
    boxplot_100ms = plt.boxplot(f1score_w100, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))

    # Boxplot per 50ms
    boxplot_50ms = plt.boxplot(f1score_w50, positions=positions[9:], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[3]))

    
    # Assegniamo i colori ai boxplot in base ai gruppi di dati
    for boxplot, color in zip([boxplot_500ms, boxplot_200ms, boxplot_100ms, boxplot_50ms], box_colors):
        for box in boxplot['boxes']:
            box.set(facecolor=color)

    plt.ylabel('F1-score (%)', fontsize=18)
    title=f'{T_or_V} Boxplot of F1-scores, windows comparison'
    #plt.title(title, fontsize=16)
    plt.ylim(ymin, 100)
    plt.yticks(fontsize=18)
    plt.xticks([])
    plt.grid(True)

    # Annotazione media e deviazione standard per ciascun boxplot
    all_f1score_data = f1score_w500 + f1score_w200 + f1score_w100 + f1score_w50
    for i, f1score_data in enumerate(all_f1score_data):
        mean = np.mean(f1score_data)
        std = np.std(f1score_data, ddof=1)
        plt.text(positions[i]-0.4, mean-2, f'{mean:.1f}\n±{std:.1f}', ha='center', va='center', fontsize=15)
        plt.text(positions[i], ymin-2, etichette[i], ha='center', va='center', fontsize=15)  # Aggiungo l'etichetta


    plt.text(2, ymin-4, '500ms', ha='center', va='center', fontsize=18)
    plt.text(6, ymin-4, '200ms', ha='center', va='center', fontsize=18)
    plt.text(10, ymin-4, '100ms', ha='center', va='center', fontsize=18)
    plt.text(14, ymin-4, '50ms', ha='center', va='center', fontsize=18)
    
    # Fill between for each window size
    for i, f1score_data in enumerate([f1score_w500, f1score_w200, f1score_w100, f1score_w50]):
        mean_values = [np.mean(data) for data in f1score_data]
        std_values = [np.std(data, ddof=1) for data in f1score_data]
        plt.fill_between(positions[i * 3:(i + 1) * 3], np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), color=box_colors[i], alpha=0.1)


    plt.savefig(f"{path_boxplots}/{title}.png")
    plt.show()

    
    
def CM_plot(base_folder,classifier_chosen, animal, window, n_classes):
    T_or_V='test_results'
    # Set the x-axis tick labels
    if n_classes==5:
        labels = ["Noci", "Dorsiflex", "Plantaflex", "Touch", "Rest"]
        dist=0.20
    else:
        labels = ["Noci", "Dorsiflex", "Plantaflex", "Touch"]
        dist=0.27
        
    # Initialize an empty list to store individual confusion matrices
    matrices = []
    
    pickle_file_folder = base_folder + animal+'/' + window+'ms/'
    
    pickle_file_path = pickle_file_folder + classifier_chosen+'_'+str(n_classes)+'_class_' + animal +'_'+ window + 'ms_results.pickle'
    
    output_cm = base_folder + 'Confusion Matrixes'
    
    if not os.path.exists(output_cm):
            os.makedirs(output_cm)
            
    print(pickle_file_path)

    # Load the pickle file
    with open(pickle_file_path, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)

        # Check if 'test_results' dictionary exists and contains 'test_confusion_matrix'
        if 'test_results' in results_dict and 'test_confusion_matrix' in results_dict['test_results']:

            # Get the list of individual confusion matrices
            matrices = results_dict[T_or_V]['test_confusion_matrix']

            # Combine individual confusion matrices and get mean and std of the K-CVs
            combined_matrix = np.sum(matrices, axis=0)

            mean_accuracy = np.mean(results_dict[T_or_V]['test_accuracy'])*100
            mean_f1_score = np.mean(results_dict[T_or_V]['test_f1_score'])*100

            std_accuracy = np.std(results_dict[T_or_V]['test_accuracy'], ddof=1)*100
            std_f1_score = np.std(results_dict[T_or_V]['test_f1_score'], ddof=1)*100

            for fold, matrix in enumerate(matrices):

                title=f"Testing - {animal} - Window {window} ms - FOLD {fold+1} of {len(matrices)}"

                f1_score=results_dict[T_or_V]['test_f1_score'][fold]*100
                accuracy=results_dict[T_or_V]['test_accuracy'][fold]*100       

                # Save the individual confusion matrix

                output_file = f'{base_folder}/{title}.png'
                plt.figure(figsize=(9,6))            
                sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', cbar=False,  annot_kws={"size": 15}, xticklabels=labels, yticklabels=labels)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

                plt.title(title, loc='left', x=0.25, fontsize=15)
                plt.xlabel('Predicted Labels', fontsize=15)
                plt.ylabel('True Labels', fontsize=15)

                # Annotate the plot with F1 score and accuracy
                plt.text(0.01, 1.06, f"F1 Score: {f1_score:.2f}%", ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
                plt.text(0.01, 1.02, f"Accuracy: {accuracy:.2f}%", ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

                output_file = output_cm +'/'+ title  # Replace with the desired output path
                plt.savefig(output_file)
                plt.show()
                plt.close()
                
            # Calculate precision, recall, and F1 score for each class
            ppv_per_class = []
            tpr_per_class = []
            f1_per_class = []
                      
            for cls in range(len(combined_matrix)):
                true_positive = combined_matrix[cls, cls]
                false_positive = np.sum(combined_matrix[:, cls]) - true_positive
                false_negative = np.sum(combined_matrix[cls, :]) - true_positive
                true_negative = np.sum(combined_matrix) - (true_positive + false_positive + false_negative)

                # Positive Predictive Value (PPV) or Precision
                ppv = true_positive / (true_positive + false_positive) *100
                ppv_per_class.append(ppv)

                # True Positive Rate (TPR) or Sensitivity or Recall
                tpr = true_positive / (true_positive + false_negative)*100
                tpr_per_class.append(tpr)

                # F1 Score
                f1 = 2 * (ppv * tpr) / (ppv + tpr)
                f1_per_class.append(f1)

            # Create a heatmap for the combined confusion matrix
            plt.figure(figsize=(9,6))
            plt.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.15)  # Adjust these values as needed

            sns.heatmap(combined_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,  annot_kws={"size": 15}, xticklabels=labels, yticklabels=labels)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('Predicted Labels', fontsize=15)
            plt.ylabel('True Labels', fontsize=15)


            title=f"Testing - {animal} - Window {window} ms - Sum of {len(matrices)} FOLDs"
            #plt.title(title, loc='left', x=0.25, fontsize=15)
            plt.text(0.2, 1.06, f"F1 Score: {mean_f1_score:.2f}%±{std_f1_score:.1f}%", ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            plt.text(0.2, 1.02, f"Accuracy: {mean_accuracy:.2f}%±{std_accuracy:.1f}%", ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)       

            for cls in range(len(combined_matrix)):
                plt.text(1.02, 0.9 - cls*dist, f"{labels[cls]}:\nPPV = {ppv_per_class[cls]:.1f}%\nTPR = {tpr_per_class[cls]:.1f}%\nF1 Score = {f1_per_class[cls]:.2f}%", ha='left', va='center', fontsize=12, transform=plt.gca().transAxes)


                #print(f"Class {cls + 1}: PPV = {ppv_per_class[cls]:.1f}, TPR = {tpr_per_class[cls]:.1f}, F1 Score = {f1_per_class[cls]:.2f}")


            # Save the combined confusion matrix as an image
            output_file = output_cm  +'/'+ title  # Replace with the desired output path
            plt.savefig(output_file)
            plt.show()
            plt.close()



#def F1_plot_perclass(base_folder,classifier_chosen, animal, window, n_classes):
   



