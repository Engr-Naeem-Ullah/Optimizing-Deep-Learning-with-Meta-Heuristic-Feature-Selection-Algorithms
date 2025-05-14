% %% Cleaning
% clear;
% clc;
% warning('off');
% 
% %% Load features and labels
% load('bestCCnetfeatureswithlables45fc8.mat'); % Ensure this file contains 'features' and 'labels'
% 
% %% Parameters for Bees Algorithm
% numBees = 30; % Number of bees
% numFeatures = size(features, 2); % Total number of features
% maxIter = 100; % Maximum number of iterations
% 
% %% Initialize Population
% population = rand(numBees, numFeatures) > 0.9; % Initialize with a sparsity
% for i = 1:numBees
%     if sum(population(i, :)) == 0
%         population(i, randi(numFeatures)) = true; % Ensure at least one feature is selected
%     end
% end
% 
% %% Evaluate Initial Fitness
% fitness = zeros(numBees, 1);
% for i = 1:numBees
%     fitness(i) = EvaluateFitness(features(:, population(i, :)), labels);
% end
% 
% %% Timing and Convergence Tracking
% convergenceHistory = zeros(maxIter, 1);
% totalTimeStart = tic; % Start timer for total execution
% 
% %% Capture initial memory use
% user = memory;
% initialMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
% maxMemoryUsed = initialMemory; % Initialize max memory usage tracker
% 
% %% Main Loop of Bees Algorithm
% featureSelectionTimeStart = tic; % Start timer for feature selection
% 
% for iter = 1:maxIter
%     % Employed Bees Phase
%     for i = 1:numBees
%         newSolution = Mutate(population(i, :));
%         newFitness = EvaluateFitness(features(:, newSolution), labels);
%         if newFitness > fitness(i)
%             population(i, :) = newSolution;
%             fitness(i) = newFitness;
%         end
%     end
% 
%     % Onlooker Bees Phase
%     probability = fitness / sum(fitness);
%     for i = 1:numBees
%         k = RouletteWheelSelection(probability);
%         newSolution = Mutate(population(k, :));
%         newFitness = EvaluateFitness(features(:, newSolution), labels);
%         if newFitness > fitness(k)
%             population(k, :) = newSolution;
%             fitness(k) = newFitness;
%         end
%     end
% 
%     % Scout Bees Phase
%     for i = 1:numBees
%         if ShouldBecomeScout(fitness(i))
%             population(i, :) = rand(1, numFeatures) > 0.5;
%             fitness(i) = EvaluateFitness(features(:, population(i, :)), labels);
%         end
%     end
% 
%     % Log best fitness of this iteration
%     convergenceHistory(iter) = max(fitness);
% 
%     % Track memory usage dynamically
%     user = memory;
%     currentMemory = user.MemUsedMATLAB / 1e6;
%     maxMemoryUsed = max(maxMemoryUsed, currentMemory);
% end
% featureSelectionTime = toc(featureSelectionTimeStart); % Stop timing for feature selection
% 
% %% Select best solution
% [bestFitness, bestIdx] = max(fitness);
% bestFeatures = population(bestIdx, :);
% selectedFeatures = features(:, bestFeatures);
% ho = 0.2; % Set validation set percentage
% HO = cvpartition(labels, 'HoldOut', ho);
% X_train = selectedFeatures(HO.training(), :);
% y_train = labels(HO.training());
% X_test = selectedFeatures(HO.test(), :);
% y_test = labels(HO.test());
% 
% classifierPerformanceTimeStart = tic; % Start timing classifier performance
% %% Classifier Models
% classifiers = {
%     @fitcknn,
%     @(X, Y) fitcecoc(X, Y, 'Learners', templateSVM('KernelFunction', 'linear'), 'Coding', 'onevsall'),
%     @fitctree,
%     @fitcnb
% };
% names = {'K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree', 'Naive Bayes'};
% 
% for i = 1:length(classifiers)
%     cvmodel = crossval(classifiers{i}(X_train, y_train), 'KFold', 5);
%     figure;
%     confusionchart(cvmodel.Y, kfoldPredict(cvmodel));
%     title([names{i} ' Confusion Matrix']);
% end
% classifierPerformanceTime = toc(classifierPerformanceTimeStart); % End timing classifier performance
% 
% totalTime = toc(totalTimeStart); % Stop timer for total execution including classifier training/testing
% 
% %% Plot Convergence History
% figure;
% plot(1:maxIter, convergenceHistory, 'b-o');
% xlabel('Iteration');
% ylabel('Best Fitness Value');
% title('Convergence History of Bees Algorithm');
% grid on;
% 
% %% Display execution times
% fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
% fprintf('Max memory used during feature selection: %.4f MB.\n', maxMemoryUsed - initialMemory);
% fprintf('Time taken for classifier training and testing: %.4f seconds.\n', classifierPerformanceTime);
% fprintf('Total execution time: %.4f seconds (includes all operations).\n', totalTime);
% 
% 
% 
% 
% %% Supporting Functions
% function fit = EvaluateFitness(subsetFeatures, labels)
%     if isempty(subsetFeatures) || size(subsetFeatures, 2) == 0
%         fit = 0;  % Assign a low fitness to encourage selection of non-empty subsets
%         return;
%     end
%     model = fitcknn(subsetFeatures, labels, 'NumNeighbors', 5);
%     cvmodel = crossval(model, 'KFold', 5);
%     fit = 1 - kfoldLoss(cvmodel);
% end
% 
% function newSol = Mutate(solution)
%     mutationRate = 0.1; % Adjust mutation rate as necessary
%     mutationPoint = randi(length(solution));
%     newSol = solution;
%     newSol(mutationPoint) = ~newSol(mutationPoint);
% end
% 
% function idx = RouletteWheelSelection(probability)
%     cumulative = cumsum(probability);
%     r = rand();
%     idx = find(cumulative >= r, 1, 'first');
% end
% 
% function scout = ShouldBecomeScout(fitness)
%     threshold = 0.7;  % Define a threshold for scouting
%     scout = fitness < threshold;
% end


%% Cleaning
clear;
clc;
warning('off');

%% Load features and labels
load('bestCCnetfeatureswithlables45fc8.mat'); % Ensure this file contains 'features' and 'labels'

%% Parameters for Bees Algorithm
numBees = 30; % Number of bees
numFeatures = size(features, 2); % Total number of features
maxIter = 100; % Maximum number of iterations

%% Initialize Population
population = rand(numBees, numFeatures) > 0.9; % Initialize with a sparsity
for i = 1:numBees
    if sum(population(i, :)) == 0
        population(i, randi(numFeatures)) = true; % Ensure at least one feature is selected
    end
end

%% Evaluate Initial Fitness
fitness = zeros(numBees, 1);
for i = 1:numBees
    fitness(i) = EvaluateFitness(features(:, population(i, :)), labels);
end

%% Timing and Convergence Tracking
convergenceHistory = zeros(maxIter, 1);
totalTimeStart = tic; % Start timer for total execution

%% Capture initial memory use
user = memory;
initialMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
maxMemoryUsed = initialMemory; % Initialize max memory usage tracker

%% Main Loop of Bees Algorithm
featureSelectionTimeStart = tic; % Start timer for feature selection

for iter = 1:maxIter
    % Employed Bees Phase
    for i = 1:numBees
        newSolution = Mutate(population(i, :));
        newFitness = EvaluateFitness(features(:, newSolution), labels);
        if newFitness > fitness(i)
            population(i, :) = newSolution;
            fitness(i) = newFitness;
        end
    end

    % Onlooker Bees Phase
    probability = fitness / sum(fitness);
    for i = 1:numBees
        k = RouletteWheelSelection(probability);
        newSolution = Mutate(population(k, :));
        newFitness = EvaluateFitness(features(:, newSolution), labels);
        if newFitness > fitness(k)
            population(k, :) = newSolution;
            fitness(k) = newFitness;
        end
    end

    % Scout Bees Phase
    for i = 1:numBees
        if ShouldBecomeScout(fitness(i))
            population(i, :) = rand(1, numFeatures) > 0.5;
            fitness(i) = EvaluateFitness(features(:, population(i, :)), labels);
        end
    end

    % Log best fitness of this iteration
    convergenceHistory(iter) = max(fitness);

    % Track memory usage dynamically
    user = memory;
    currentMemory = user.MemUsedMATLAB / 1e6;
    maxMemoryUsed = max(maxMemoryUsed, currentMemory);
end
featureSelectionTime = toc(featureSelectionTimeStart); % Stop timing for feature selection

%% Select best solution
[bestFitness, bestIdx] = max(fitness);
bestFeatures = population(bestIdx, :);
selectedFeatures = features(:, bestFeatures);
ho = 0.2; % Set validation set percentage
HO = cvpartition(labels, 'HoldOut', ho);
X_train = selectedFeatures(HO.training(), :);
y_train = labels(HO.training());
X_test = selectedFeatures(HO.test(), :);
y_test = labels(HO.test());

% Store results for number of features and classification accuracy
classifierNames = {'K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree', 'Naive Bayes'};
numFeaturesUsed = zeros(1, 4);  % To store the number of features used for each classifier
classificationResults = zeros(1, 4);  % To store the classification accuracy for each classifier

classifierPerformanceTimeStart = tic; % Start timing classifier performance

%% Classifier Models
classifiers = {
    @fitcknn,
    @(X, Y) fitcecoc(X, Y, 'Learners', templateSVM('KernelFunction', 'linear'), 'Coding', 'onevsall'),
    @fitctree,
    @fitcnb
};

for i = 1:length(classifiers)
    % Train the classifier
    model = classifiers{i}(X_train, y_train);
    pred = predict(model, X_test);
    
    % Calculate confusion matrix
    cm = confusionmat(y_test, pred);
    
    % Store the number of features used and classification accuracy
    numFeaturesUsed(i) = sum(bestFeatures);  % Number of features used by the current classifier
    classificationResults(i) = sum(pred == y_test) / length(y_test);  % Classification accuracy
    
    % Plot confusion matrix
    figure;
    confusionchart(cm);
    title([classifierNames{i} ' Confusion Matrix']);
end
classifierPerformanceTime = toc(classifierPerformanceTimeStart); % End timing classifier performance

totalTime = toc(totalTimeStart); % Stop timer for total execution including classifier training/testing

%% Plot Convergence History
figure;
plot(1:maxIter, convergenceHistory, 'b-o');
xlabel('Iteration');
ylabel('Best Fitness Value');
title('Convergence History of Bees Algorithm');
grid on;

%% Display execution times
fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
fprintf('Max memory used during feature selection: %.4f MB.\n', maxMemoryUsed - initialMemory);
fprintf('Time taken for classifier training and testing: %.4f seconds.\n', classifierPerformanceTime);
fprintf('Total execution time: %.4f seconds (includes all operations).\n', totalTime);

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


%% Supporting Functions
function fit = EvaluateFitness(subsetFeatures, labels)
    if isempty(subsetFeatures) || size(subsetFeatures, 2) == 0
        fit = 0;  % Assign a low fitness to encourage selection of non-empty subsets
        return;
    end
    model = fitcknn(subsetFeatures, labels, 'NumNeighbors', 5);
    cvmodel = crossval(model, 'KFold', 5);
    fit = 1 - kfoldLoss(cvmodel);
end

function newSol = Mutate(solution)
    mutationRate = 0.1; % Adjust mutation rate as necessary
    mutationPoint = randi(length(solution));
    newSol = solution;
    newSol(mutationPoint) = ~newSol(mutationPoint);
end

function idx = RouletteWheelSelection(probability)
    cumulative = cumsum(probability);
    r = rand();
    idx = find(cumulative >= r, 1, 'first');
end

function scout = ShouldBecomeScout(fitness)
    threshold = 0.7;  % Define a threshold for scouting
    scout = fitness < threshold;
end
