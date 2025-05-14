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
    mutationPoint = randi(length(solution));
    newSol = solution;
    if sum(solution) == 1 && solution(mutationPoint) == true
        % If only one feature is selected and it is the mutation point,
        % flip another feature to true instead
        newMutationPoint = randi(length(solution));
        while newMutationPoint == mutationPoint
            newMutationPoint = randi(length(solution));
        end
        newSol(newMutationPoint) = true;
    else
        newSol(mutationPoint) = ~newSol(mutationPoint);
    end
end


function idx = RouletteWheelSelection(probability)
    % Roulette Wheel Selection
    cumulative = cumsum(probability);
    r = rand();
    idx = find(cumulative >= r, 1, 'first');
end

function scout = ShouldBecomeScout(fitness)
    % Determine if a bee should become a scout
    scout = fitness < 0.7;  % Example threshold
end
