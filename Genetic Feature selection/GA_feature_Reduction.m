% %% Majid Farzaneh
% %% Genetic Algorithm for Feature Selection in Classification Problems
% clc;
% clear;
% close all;
% 
% %% Load dataset and set global variables
% load bestCCnetfeatureswithlables.mat; % This file should contain 'features' and 'labels'
% whos
% global orgfeatures labels alg
% orgfeatures = features; % Assuming 'features' is loaded from the .mat file
% labels = labels; % Assuming 'labels' is loaded from the .mat file
% alg = 'KNN'; % Example: change as needed to 'NB', 'DT', 'SVM', etc.
% 
% %% Initialization
% npop = 10; % Population size
% max_generation = 50; % Reducing for quicker results, adjust as needed
% Nf = size(orgfeatures, 2); % Number of features
% 
% % Initialize population with random solutions
% population = randi([0 1], npop, Nf);
% 
% % Fitness array
% fitness = zeros(npop, 1);
% 
% % Total Execution Timer
% totalExecutionStart = tic;
% 
% % Capture initial memory use
% user = memory;
% initialMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
% 
% %% Genetic Algorithm Operations
% for gen = 1:max_generation
%     % Evaluate fitness
%     for i = 1:npop
%         fitness(i) = fitf(logical(population(i, :))); % Adjust to use global variable setup
%     end
% 
%     % Genetic operators
%     % Selection based on fitness
%     [~, sortedIdx] = sort(fitness, 'descend');
%     population = population(sortedIdx, :); % Keep best solutions
% 
%     % Crossover (single point)
%     for i = 1:2:npop-1
%         point = randi([1 Nf-1]);
%         population(i+1,:) = [population(i,1:point) population(i+1,point+1:end)];
%     end
% 
%     % Mutation
%     for i = 1:npop
%         if rand < 0.1 % Mutation probability
%             point = randi([1 Nf]);
%             population(i, point) = ~population(i, point);
%         end
%     end
% 
%     % Display progress
%     if mod(gen, 10) == 0
%         disp(['Generation ', num2str(gen), ' Best Fitness: ', num2str(max(fitness))]);
%     end
% end
% 
% % Feature Selection Timer
% featureSelectionTime = toc(totalExecutionStart);
% 
% % Calculate memory used after the algorithm run
% user = memory;
% finalMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
% memoryUsed = finalMemory - initialMemory;
% 
% %% Use the best solution
% [bestFitness, bestIdx] = max(fitness);
% bestSolution = logical(population(bestIdx, :));
% if sum(bestSolution) == 0
%     error('No features were selected. Adjust the genetic algorithm settings.');
% end
% selectedFeatures = orgfeatures(:, bestSolution);
% 
% %% Split dataset into training and testing using hold-out
% ho = 0.2; % Set validation set percentage
% HO = cvpartition(labels, 'HoldOut', ho);
% X_train = selectedFeatures(HO.training(), :);
% y_train = labels(HO.training());
% X_test = selectedFeatures(HO.test(), :);
% y_test = labels(HO.test());
% 
% % Check if there's any data to train on
% if isempty(X_train) || isempty(X_test)
%     error('Selected feature set resulted in empty training or testing sets.');
% end
% 
% %% Train and Test Classifiers
% classifierPerformanceTimeStart = tic; % Start timing classifier performance
% 
% % KNN
% modelKNN = fitcknn(X_train, y_train);
% predKNN = predict(modelKNN, X_test);
% CKNN = confusionmat(y_test, predKNN);
% 
% % SVM with ECOC for multiclass classification
% modelSVM = fitcecoc(X_train, y_train);
% predSVM = predict(modelSVM, X_test);
% CSVM = confusionmat(y_test, predSVM);
% 
% % Naive Bayes
% modelNB = fitcnb(X_train, y_train);
% predNB = predict(modelNB, X_test);
% CNB = confusionmat(y_test, predNB);
% 
% % Decision Tree
% modelTree = fitctree(X_train, y_train);
% predTree = predict(modelTree, X_test);
% CTree = confusionmat(y_test, predTree);
% 
% classifierPerformanceTime = toc(classifierPerformanceTimeStart); % End timing classifier performance
% 
% % Total Execution Time
% totalExecutionTime = toc(totalExecutionStart);
% 
% %% Display Results
% fprintf('\nConfusion Matrix for KNN:\n');
% disp(CKNN);
% fprintf('\nConfusion Matrix for SVM (ECOC):\n');
% disp(CSVM);
% fprintf('\nConfusion Matrix for Naive Bayes:\n');
% disp(CNB);
% fprintf('\nConfusion Matrix for Decision Tree:\n');
% disp(CTree);
% 
% %% Save the selected features and models
% save('GA_SelectedFeaturesModels.mat', 'modelKNN', 'modelSVM', 'modelNB', 'modelTree', 'selectedFeatures');
% 
% %% Display execution times and memory usage
% fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
% fprintf('Memory used for feature selection: %.4f MB.\n', memoryUsed);
% fprintf('Time taken for classifier training and testing: %.4f seconds.\n', classifierPerformanceTime);
% fprintf('Total execution time: %.4f seconds.\n', totalExecutionTime);


%% Majid Farzaneh
%% Genetic Algorithm for Feature Selection in Classification Problems
clc;
clear;
close all;

%% Load dataset and set global variables
load bestCCnetfeatureswithlables.mat; % This file should contain 'features' and 'labels'
whos
global orgfeatures labels alg
orgfeatures = features; % Assuming 'features' is loaded from the .mat file
labels = labels; % Assuming 'labels' is loaded from the .mat file
alg = 'KNN'; % Example: change as needed to 'NB', 'DT', 'SVM', etc.

%% Initialization
npop = 10; % Population size
max_generation = 50; % Reducing for quicker results, adjust as needed
Nf = size(orgfeatures, 2); % Number of features

% Initialize population with random solutions
population = randi([0 1], npop, Nf);

% Fitness array
fitness = zeros(npop, 1);

% Total Execution Timer
totalExecutionStart = tic;

% Capture initial memory use
user = memory;
initialMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB

%% Genetic Algorithm Operations
for gen = 1:max_generation
    % Evaluate fitness
    for i = 1:npop
        fitness(i) = fitf(logical(population(i, :))); % Adjust to use global variable setup
    end

    % Genetic operators
    % Selection based on fitness
    [~, sortedIdx] = sort(fitness, 'descend');
    population = population(sortedIdx, :); % Keep best solutions

    % Crossover (single point)
    for i = 1:2:npop-1
        point = randi([1 Nf-1]);
        population(i+1,:) = [population(i,1:point) population(i+1,point+1:end)];
    end

    % Mutation
    for i = 1:npop
        if rand < 0.1 % Mutation probability
            point = randi([1 Nf]);
            population(i, point) = ~population(i, point);
        end
    end

    % Display progress
    if mod(gen, 10) == 0
        disp(['Generation ', num2str(gen), ' Best Fitness: ', num2str(max(fitness))]);
    end
end

% Feature Selection Timer
featureSelectionTime = toc(totalExecutionStart);

% Calculate memory used after the algorithm run
user = memory;
finalMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
memoryUsed = finalMemory - initialMemory;

%% Use the best solution
[bestFitness, bestIdx] = max(fitness);
bestSolution = logical(population(bestIdx, :));
if sum(bestSolution) == 0
    error('No features were selected. Adjust the genetic algorithm settings.');
end
selectedFeatures = orgfeatures(:, bestSolution);

%% Split dataset into training and testing using hold-out
ho = 0.2; % Set validation set percentage
HO = cvpartition(labels, 'HoldOut', ho);
X_train = selectedFeatures(HO.training(), :);
y_train = labels(HO.training());
X_test = selectedFeatures(HO.test(), :);
y_test = labels(HO.test());

% Check if there's any data to train on
if isempty(X_train) || isempty(X_test)
    error('Selected feature set resulted in empty training or testing sets.');
end

%% Train and Test Classifiers
classifierPerformanceTimeStart = tic; % Start timing classifier performance

% Store results for number of features and classification accuracy
classifierNames = {'KNN', 'SVM', 'Naive Bayes', 'Decision Tree'};
numFeaturesUsed = zeros(1, 4);  % To store the number of features used for each classifier
classificationResults = zeros(1, 4);  % To store the classification accuracy for each classifier

% KNN
modelKNN = fitcknn(X_train, y_train);
predKNN = predict(modelKNN, X_test);
CKNN = confusionmat(y_test, predKNN);
numFeaturesUsed(1) = sum(bestSolution);  % Number of features used by KNN
classificationResults(1) = sum(predKNN == y_test) / length(y_test);

% SVM with ECOC for multiclass classification
modelSVM = fitcecoc(X_train, y_train);
predSVM = predict(modelSVM, X_test);
CSVM = confusionmat(y_test, predSVM);
numFeaturesUsed(2) = sum(bestSolution);  % Number of features used by SVM
classificationResults(2) = sum(predSVM == y_test) / length(y_test);

% Naive Bayes
modelNB = fitcnb(X_train, y_train);
predNB = predict(modelNB, X_test);
CNB = confusionmat(y_test, predNB);
numFeaturesUsed(3) = sum(bestSolution);  % Number of features used by Naive Bayes
classificationResults(3) = sum(predNB == y_test) / length(y_test);

% Decision Tree
modelTree = fitctree(X_train, y_train);
predTree = predict(modelTree, X_test);
CTree = confusionmat(y_test, predTree);
numFeaturesUsed(4) = sum(bestSolution);  % Number of features used by Decision Tree
classificationResults(4) = sum(predTree == y_test) / length(y_test);

classifierPerformanceTime = toc(classifierPerformanceTimeStart); % End timing classifier performance

% Total Execution Time
totalExecutionTime = toc(totalExecutionStart);

%% Display Results
fprintf('\nConfusion Matrix for KNN:\n');
disp(CKNN);
fprintf('\nConfusion Matrix for SVM (ECOC):\n');
disp(CSVM);
fprintf('\nConfusion Matrix for Naive Bayes:\n');
disp(CNB);
fprintf('\nConfusion Matrix for Decision Tree:\n');
disp(CTree);

%% Save the selected features and models
save('GA_SelectedFeaturesModels.mat', 'modelKNN', 'modelSVM', 'modelNB', 'modelTree', 'selectedFeatures');

%% Display execution times and memory usage
fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
fprintf('Memory used for feature selection: %.4f MB.\n', memoryUsed);
fprintf('Time taken for classifier training and testing: %.4f seconds.\n', classifierPerformanceTime);
fprintf('Total execution time: %.4f seconds.\n', totalExecutionTime);

%% Plot the relationship between the number of features and classification performance
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
