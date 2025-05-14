% %% Particle Swarm Optimization with Multiple Classifiers
% clc; clear; close all; 
% 
% % Load dataset
% load bestCCnetfeatureswithlables45fc8.mat; 
% 
% % Set validation set percentage
% ho = 0.2; 
% 
% % Hold-out method for splitting the dataset
% HO = cvpartition(labels,'HoldOut',ho);
% 
% % PSO parameter settings
% N        = 10;
% max_Iter = 100;
% c1       = 2;     % Cognitive factor
% c2       = 2;     % Social factor
% w        = 1;     % Inertia weight
% 
% % Total Execution Timer
% totalExecutionStart = tic;
% 
% % Execute Particle Swarm Optimization
% featureSelectionStart = tic; % Start feature selection timer
% [sFeat, Sf, Nf, curve] = jPSO(features, labels, N, max_Iter, c1, c2, w, HO);
% featureSelectionTime = toc(featureSelectionStart); % Stop feature selection timer
% 
% % Display the selected features and indices
% disp('Selected Features:');
% disp(sFeat);
% disp('Selected Feature Indices:');
% disp(Sf);
% disp(['Number of Selected Features: ', num2str(Nf)]);
% 
% % Using only selected features
% selectedFeatures = features(:, Sf);
% 
% % KNN
% modelKNN = fitcknn(selectedFeatures(HO.training,:), labels(HO.training));
% predKNN = predict(modelKNN, selectedFeatures(HO.test,:));
% 
% % SVM with ECOC
% modelSVM = fitcecoc(selectedFeatures(HO.training,:), labels(HO.training));
% predSVM = predict(modelSVM, selectedFeatures(HO.test,:));
% 
% % Naive Bayes
% modelNB = fitcnb(selectedFeatures(HO.training,:), labels(HO.training));
% predNB = predict(modelNB, selectedFeatures(HO.test,:));
% 
% % Decision Tree
% modelTree = fitctree(selectedFeatures(HO.training,:), labels(HO.training));
% predTree = predict(modelTree, selectedFeatures(HO.test,:));
% 
% % Calculate confusion matrices
% CKNN = confusionmat(labels(HO.test), predKNN);
% CSVM = confusionmat(labels(HO.test), predSVM);
% CNB = confusionmat(labels(HO.test), predNB);
% CTree = confusionmat(labels(HO.test), predTree);
% 
% % Display confusion matrices
% disp('Confusion Matrix KNN:');
% disp(CKNN);
% disp('Confusion Matrix SVM:');
% disp(CSVM);
% disp('Confusion Matrix Naive Bayes:');
% disp(CNB);
% disp('Confusion Matrix Decision Tree:');
% disp(CTree);
% 
% % Plot convergence curve
% figure;
% plot(1:max_Iter, curve);
% xlabel('Number of iterations');
% ylabel('Fitness Value');
% title('PSO Convergence Curve');
% grid on;
% 
% % Save models, optional
% save('PSO_Models.mat', 'modelKNN', 'modelSVM', 'modelNB', 'modelTree');
% 
% % Total Execution Timer
% totalExecutionTime = toc(totalExecutionStart);
% 
% % Display execution times
% fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
% fprintf('Total execution time: %.4f seconds.\n', totalExecutionTime);
%% Particle Swarm Optimization with Multiple Classifiers
clc; clear; close all; 

% Load dataset
load bestCCnetfeatureswithlables45fc8.mat; 

% Set validation set percentage
ho = 0.2; 

% Hold-out method for splitting the dataset
HO = cvpartition(labels, 'HoldOut', ho);

% PSO parameter settings
N        = 10;
max_Iter = 100;
c1       = 2;     % Cognitive factor
c2       = 2;     % Social factor
w        = 1;     % Inertia weight

% Total Execution Timer
totalExecutionStart = tic;

% Execute Particle Swarm Optimization
featureSelectionStart = tic; % Start feature selection timer
[sFeat, Sf, Nf, curve] = jPSO(features, labels, N, max_Iter, c1, c2, w, HO);
featureSelectionTime = toc(featureSelectionStart); % Stop feature selection timer

% Display the selected features and indices
disp('Selected Features:');
disp(sFeat);
disp('Selected Feature Indices:');
disp(Sf);
disp(['Number of Selected Features: ', num2str(Nf)]);

% Using only selected features
selectedFeatures = features(:, Sf);

% Initialize variables to track classifier performance
classifierNames = {'KNN', 'SVM', 'Naive Bayes', 'Decision Tree'};
numFeaturesUsed = zeros(1, 4);  % To store the number of features used for each classifier
classificationResults = zeros(1, 4);  % To store the classification accuracy for each classifier

% KNN
modelKNN = fitcknn(selectedFeatures(HO.training,:), labels(HO.training));
predKNN = predict(modelKNN, selectedFeatures(HO.test,:));
numFeaturesUsed(1) = sum(Sf);  % Number of features used by KNN
classificationResults(1) = sum(predKNN == labels(HO.test)) / length(labels(HO.test));

% SVM with ECOC
modelSVM = fitcecoc(selectedFeatures(HO.training,:), labels(HO.training));
predSVM = predict(modelSVM, selectedFeatures(HO.test,:));
numFeaturesUsed(2) = sum(Sf);  % Number of features used by SVM
classificationResults(2) = sum(predSVM == labels(HO.test)) / length(labels(HO.test));

% Naive Bayes
modelNB = fitcnb(selectedFeatures(HO.training,:), labels(HO.training));
predNB = predict(modelNB, selectedFeatures(HO.test,:));
numFeaturesUsed(3) = sum(Sf);  % Number of features used by Naive Bayes
classificationResults(3) = sum(predNB == labels(HO.test)) / length(labels(HO.test));

% Decision Tree
modelTree = fitctree(selectedFeatures(HO.training,:), labels(HO.training));
predTree = predict(modelTree, selectedFeatures(HO.test,:));
numFeaturesUsed(4) = sum(Sf);  % Number of features used by Decision Tree
classificationResults(4) = sum(predTree == labels(HO.test)) / length(labels(HO.test));

% Calculate confusion matrices
CKNN = confusionmat(labels(HO.test), predKNN);
CSVM = confusionmat(labels(HO.test), predSVM);
CNB = confusionmat(labels(HO.test), predNB);
CTree = confusionmat(labels(HO.test), predTree);

% Display confusion matrices
disp('Confusion Matrix KNN:');
disp(CKNN);
disp('Confusion Matrix SVM:');
disp(CSVM);
disp('Confusion Matrix Naive Bayes:');
disp(CNB);
disp('Confusion Matrix Decision Tree:');
disp(CTree);

% Plot convergence curve for PSO
figure;
plot(1:max_Iter, curve);
xlabel('Number of iterations');
ylabel('Fitness Value');
title('PSO Convergence Curve');
grid on;

% Save models, optional
save('PSO_Models.mat', 'modelKNN', 'modelSVM', 'modelNB', 'modelTree');

% Total Execution Timer
totalExecutionTime = toc(totalExecutionStart);

% Display execution times
fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
fprintf('Total execution time: %.4f seconds.\n', totalExecutionTime);

% Plot the relationship between the number of features and classification performance
figure;
bar(classificationResults); % Using classification results (accuracy) as y-values
set(gca, 'XTickLabel', classifierNames); % X-axis as classifiers
xlabel('Classifier');
ylabel('Accuracy');
title('Classification Performance vs. Number of Features');
grid on;

% Display the relationship between number of features and classifier performance
disp('Number of features used by each classifier:');
for i = 1:length(classifierNames)
    disp([classifierNames{i} ': ' num2str(numFeaturesUsed(i)) ' features']);
end

disp('Classification performance (accuracy) for each classifier:');
for i = 1:length(classifierNames)
    disp([classifierNames{i} ': Accuracy = ' num2str(classificationResults(i)) ', Features used = ' num2str(numFeaturesUsed(i))]);
end
