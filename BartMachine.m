classdef BartMachine
    %BARTMACHINE Translation of the R interface for the bartMachine package

    properties (Constant)
        BART_NUM_CORES_DEFAULT = 1
        COLORS = rand([3,500]) .* 0.7;
    end
    properties
        java_bart_machine
        training_data_features
        training_data_features_with_missing_features
        X
        y
        y_levels
        pred_type
        model_matrix_training_data
        n
        p
        num_cores
        num_trees
        num_burn_in
        num_iterations_after_burn_in
        num_gibbs
        alpha
        beta
        k
        q
        nu
        prob_rule_class
        mh_prob_steps
        s_sq_y
        run_in_sample
        sig_sq_est
        time_to_build
        cov_prior_vec
        interaction_constraints
        use_missing_data
        use_missing_data_dummies_as_covars
        replace_missing_data_with_x_j_bar
        impute_missingness_with_rf_impute
        impute_missingness_with_x_j_bar_for_lm
        verbose
        serialize
        mem_cache_for_speed
        flush_indices_to_save_RAM
        debug_log
        seed
        num_rand_samps_in_library
        y_hat_train
        p_hat_train
        residuals
        L1_err_train
        L2_err_train
        PseudoRsq
        rmse_train
        confusion_matrix
        misclassification_error
    end

    methods
        function this = BartMachine(named_args)
            arguments
                named_args.X {mustBeTable} = table()
                named_args.y = []
                named_args.Xy = table()
                named_args.num_trees (1,1) {mustBeInteger, mustBePositive} = 50
                named_args.num_burn_in (1,1) {mustBeInteger, mustBePositive} = 250
                named_args.num_iterations_after_burn_in (1,1) {mustBeInteger, mustBePositive} = 1000
                named_args.alpha (1,1) = 0.95
                named_args.beta (1,1) = 2
                named_args.k (1,1) = 2
                named_args.q (1,1) = 0.9
                named_args.nu (1,1) = 3.0
                named_args.prob_rule_class(1,1) = 0.5
                named_args.mh_prob_steps {mustBeNumeric} = [2.5, 2.5, 4] / 9
                named_args.debug_log (1,1) {mustBeLogical} = false
                named_args.run_in_sample (1,1) {mustBeLogical} = true
                named_args.s_sq_y string {mustBeMember(named_args.s_sq_y, ["mse", "var"])} = "mse"
                named_args.sig_sq_est = []
                named_args.print_tree_illustrations (1,1) {mustBeLogical} = false
                named_args.cov_prior_vec = []
                named_args.interaction_constraints = []
                named_args.use_missing_data (1,1) {mustBeLogical} = false
                named_args.covariates_to_permute = []
                named_args.num_rand_samps_in_library (1,1) {mustBeInteger, mustBePositive} = 10000
                named_args.use_missing_data_dummies_as_covars (1,1) {mustBeLogical} = false
                named_args.replace_missing_data_with_x_j_bar (1,1) {mustBeLogical} = false
                named_args.impute_missingness_with_rf_impute (1,1) {mustBeLogical} = false
                named_args.impute_missingness_with_x_j_bar_for_lm (1,1) {mustBeLogical} = true
                named_args.mem_cache_for_speed (1,1) {mustBeLogical} = true
                named_args.flush_indices_to_save_RAM (1,1) {mustBeLogical} = true
                named_args.serialize (1,1) {mustBeLogical} = false
                named_args.seed int32 = []
                named_args.verbose (1,1) {mustBeLogical} = true
            end
            % Matlab classes must be constructible with no arguments
            if isempty(named_args.X)
                return
            end
            X = named_args.X;
            y = named_args.y;
            Xy = named_args.Xy;
            num_trees = named_args.num_trees;
            num_burn_in = named_args.num_burn_in;
            num_iterations_after_burn_in = named_args.num_iterations_after_burn_in;
            alpha = named_args.alpha;
            beta = named_args.beta;
            k = named_args.k;
            q = named_args.q;
            nu = named_args.nu;
            prob_rule_class = named_args.prob_rule_class;
            mh_prob_steps = named_args.mh_prob_steps;
            debug_log = named_args.debug_log;
            run_in_sample = named_args.run_in_sample;
            s_sq_y = named_args.s_sq_y;
            sig_sq_est = named_args.sig_sq_est;
            print_tree_illustrations = named_args.print_tree_illustrations;
            cov_prior_vec = named_args.cov_prior_vec;
            interaction_constraints = named_args.interaction_constraints;
            use_missing_data = named_args.use_missing_data;
            covariates_to_permute = named_args.covariates_to_permute;
            num_rand_samps_in_library = named_args.num_rand_samps_in_library;
            use_missing_data_dummies_as_covars = named_args.use_missing_data_dummies_as_covars;
            replace_missing_data_with_x_j_bar = named_args.replace_missing_data_with_x_j_bar;
            impute_missingness_with_rf_impute = named_args.impute_missingness_with_rf_impute;
            impute_missingness_with_x_j_bar_for_lm = named_args.impute_missingness_with_x_j_bar_for_lm;
            mem_cache_for_speed = named_args.mem_cache_for_speed;
            flush_indices_to_save_RAM = named_args.flush_indices_to_save_RAM;
            serialize = named_args.serialize;
            seed = named_args.seed;
            verbose = named_args.verbose;

            if (verbose)
                disp(['bartMachine initializing with ',num2str(num_trees),' trees...']);
            end
            t0 = tic;

            if (use_missing_data_dummies_as_covars && replace_missing_data_with_x_j_bar)
                error('You cannot impute by averages and use missing data as dummies simultaneously.');
            end
            if ((isempty(X) && isempty(Xy)) || (isempty(y) && isempty(Xy)))
                error('You need to give bartMachine a training set either by specifying X and y or by specifying a matrix Xy which contains the response named "y."');
            elseif (~isempty(X) && ~isempty(y) && ~isempty(Xy))
                error('You cannot specify both X,y and Xy simultaneously.');
            elseif (isempty(X) && isempty(y)) %they specified Xy, so now just pull out X,y
                %first ensure it's a table
                if ~istable(Xy)
                    error('The training data Xy must be a table.');
                end
                y = Xy(:, end);
                for cov = 1:(size(Xy, 2) - 1)
                    if isempty(Xy.Properties.VariableNames{cov})
                        Xy.Properties.VariableNames{cov} = ['V', num2str(cov)];
                    end
                end
                X = Xy(:, 1:(size(Xy, 2) - 1));
                X.Properties.VariableNames = Xy.Properties.VariableNames(1:(size(Xy, 2) - 1));
            end

            %make sure it's a table
            if ~istable(X)
                error('The training data X must be a table.');
            end
            if (verbose)
                disp('bartMachine vars checked...');
            end


            % we are about to construct a bartMachine object.
            % now take care of classification or regression
            y_levels = unique(y);
            if isnumeric(y) % if y is numeric, then it's a regression problem
                % java expects doubles, not ints, so we need to cast this now to avoid errors later
                y = double(y);
                java_bart_machine = bartMachine.bartMachineRegressionMultThread;
                y_remaining = y;
                pred_type = "regression";
                if isinteger(y)
                    disp("Warning: The response y is integer, bartMachine will run regression.")
                end
            elseif iscategorical(y) && length(y_levels) == 2 % if y is a categorical variable and binary
                java_bart_machine = bartMachine.bartMachineClassificationMultThread;
                y_remaining = double(y == y_levels(1));
                pred_type = "classification";
            else % otherwise throw an error
                error("Your response must be either numeric or a categorical variable with two levels.")
            end

            num_gibbs = num_burn_in + num_iterations_after_burn_in;

            if size(X, 2) == 0
                error("Your data matrix must have at least one attribute.")
            end
            if size(X, 1) == 0
                error("Your data matrix must have at least one observation.")
            end
            if length(y) ~= size(X, 1)
                error("The number of responses must be equal to the number of observations in the training data.")
            end
            if verbose
                disp("bartMachine java init...")
            end

            % if no column names, make up names
            if isempty(X.Properties.VariableNames)
                X.Properties.VariableNames = strcat("V", string(1:size(X, 2)));
            end

            if any(mh_prob_steps < 0)
                error("The grow, prune, change ratio parameter vector must all be greater than 0.")
            end



            % regenerate factors for factor columns
            predictors_which_are_factors = X.Properties.VariableNames;
            for i=1:numel(predictors_which_are_factors)
                if iscell(X.(predictors_which_are_factors{i}))
                    X.(predictors_which_are_factors{i}) = categorical(X.(predictors_which_are_factors{i}));
                end
            end

            if sum(isnan(y_remaining)) > 0
                error("You cannot have any missing data in your response vector.");
            end

            rf_imputations_for_missing = [];
            if impute_missingness_with_rf_impute
                if sum(any(isnan(X))) == 0
                    warning("No missing entries in the training data to impute.");
                    rf_imputations_for_missing = X;
                else
                    % use cols that have missing data
                    predictor_colnums_with_missingness = find(any(isnan(X)));

                    % use rfImpute for missing values
                    rf_imputations_for_missing = rfImpute(X, y);
                    rf_imputations_for_missing(:,1) = [];
                    rf_imputations_for_missing = rf_imputations_for_missing(:, predictor_colnums_with_missingness);
                end
                % rename columns of rf_imputations_for_missing
                rf_imputations_for_missing.Properties.VariableNames = strcat(rf_imputations_for_missing.Properties.VariableNames, '_imp');
                if verbose
                    disp("bartMachine after rf imputations...");
                end
            end

            % get rid of missing data if use_missing_data and replace_missing_data_with_x_j_bar are not used
            if ~use_missing_data && ~replace_missing_data_with_x_j_bar
                rows_before = size(X, 1);
                X = rmmissing(X);
                rows_after = size(X, 1);
                if rows_before - rows_after > 0
                    error(sprintf("You have %d observations with missing data. \nYou must either omit your missing data using ""rmmissing()"" or turn on the\n""use_missing_data"" or ""replace_missing_data_with_x_j_bar"" feature in order to use bartMachine.\n", rows_before - rows_after));
                end
            elseif replace_missing_data_with_x_j_bar
                X = BartMachine.imputeMatrixByXbarjContinuousOrModalForBinary(X, X);
                if verbose
                    disp("Imputed missing data using attribute averages.");
                end
            end

            if verbose
                disp("bartMachine before preprocess...");
            end
            

            pre_process_obj = BartMachine.pre_process_training_data(X, use_missing_data_dummies_as_covars, rf_imputations_for_missing);
            model_matrix_training_data = pre_process_obj.data;
            if ismember("y_", model_matrix_training_data.Properties.VariableNames)
                error("y_ cannot be used as a feature name");
            end
            model_matrix_training_data.y_ = y_remaining; % Tag on response with hopefully unused name
            p = size(model_matrix_training_data, 2) - 1; % we subtract one because we tacked on the response as the last column
            factor_lengths = pre_process_obj.factor_lengths;
            if verbose
                fprintf('bartMachine after preprocess... %d total features...\n', p);
            end

            % now create a default cov_prior_vec that factors in the levels of the factors
            null_cov_prior_vec = isempty(cov_prior_vec);
            if null_cov_prior_vec && ~isempty(factor_lengths)
                % begin with the uniform
                cov_prior_vec = ones(1, p);
                j_factor_begin = p - sum(factor_lengths) + 1;
                for l = 1:length(factor_lengths)
                    factor_length = factor_lengths(l);
                    cov_prior_vec(j_factor_begin:j_factor_begin+factor_length-1) = 1 / factor_length;
                    j_factor_begin = j_factor_begin + factor_length;
                end
            end

            if ~isempty(interaction_constraints)
                if ~mem_cache_for_speed
                    error('In order to use interaction constraints, "mem_cache_for_speed" must be set to TRUE.');
                end
                if ~iscell(interaction_constraints)
                    error('specified parameter "interaction_constraints" must be a list');
                elseif isempty(interaction_constraints)
                    error('interaction_constraints list cannot be empty');
                end
                for a = 1:length(interaction_constraints)
                    vars_a = interaction_constraints{a};
                    % check if the constraint components are valid features
                    for b = 1:length(vars_a)
                        var_elem = vars_a(b);
                        if isnumeric(var_elem) || isinteger(var_elem)
                            if ~(var_elem >= 1 && var_elem <= p)
                                error(['Element ' num2str(var_elem) ' in interaction_constraints vector number ' num2str(a) ' is numeric but not one of 1, ..., ' num2str(p) ', where ' num2str(p) ' is the number of columns in X.']);
                            end
                        end
                        if isstring(var_elem) || ischar(var_elem)
                            if ~ismember(var_elem, X.Properties.VariableNames)
                                error(['Element ' var_elem ' in interaction_constraints vector number ' num2str(a) ' is a string but not one of the column names of X.']);
                            end
                        end
                        % force all it be integers and begin index at zero
                        if isnumeric(var_elem) || isinteger(var_elem)
                            vars_a(b) = var_elem - 1;
                        elseif isstring(var_elem) || ischar(var_elem)
                            vars_a(b) = find(strcmp(X.Properties.VariableNames, var_elem)) - 1;
                        end
                    end
                    interaction_constraints{a} = int32(vars_a);
                end
            end



            if ~isempty(covariates_to_permute)
                % first check if these covariates are even in the matrix to begin with
                for cov = covariates_to_permute
                    if ~any(strcmp(cov, model_matrix_training_data.Properties.VariableNames))
                        error(['Covariate "', cov, '" not found in design matrix.'])
                    end
                end
                permuted_order = randperm(size(model_matrix_training_data, 1));
                model_matrix_training_data{:, covariates_to_permute} = model_matrix_training_data{permuted_order, covariates_to_permute};
            end

            % now set whether we want the program to log to a file
            if debug_log && verbose
                warning('printing out the log file will slow down the runtime significantly.')
                java_bart_machine.writeStdOutToLogFile;
            end

            % set whether we want there to be tree illustrations
            if print_tree_illustrations
                warning('printing tree illustrations is excruciatingly slow.')
                java_bart_machine.printTreeIllustations;
            end

            % set the std deviation of y to use
            if size(model_matrix_training_data, 2) - 1 >= size(model_matrix_training_data, 1)
                if verbose
                    warning('cannot use MSE of linear model for s_sq_y if p > n. bartMachine will use sample var(y) instead.')
                end
                s_sq_y = 'var';
            end

        


            %estimate sigma^2 to be given to the BART model
            if isempty(sig_sq_est)
                if strcmp(pred_type, 'regression')
                    y_range = max(y) - min(y);
                    y_trans = (y - min(y)) / y_range - 0.5;
                    if strcmp(s_sq_y, 'mse')
                        X_for_lm = model_matrix_training_data(:, 1:end-1);
                        if impute_missingness_with_x_j_bar_for_lm
                            X_for_lm = BartMachine.imputeMatrixByXbarjContinuousOrModalForBinary(X_for_lm, X_for_lm);
                        elseif isempty(naomit(X_for_lm))
                            error('The data does not have enough full records to estimate a naive prediction error. Please rerun with "impute_missingness_with_x_j_bar_for_lm" set to true.');
                        end
                        mod = fitlm(X_for_lm{:,:}, y_trans);
                        mse = var(mod.Residuals.Raw);
                        sig_sq_est = mse;
                        java_bart_machine.setSampleVarY(sig_sq_est);
                    elseif strcmp(s_sq_y, 'var')
                        sig_sq_est = var(y_trans);
                        java_bart_machine.setSampleVarY(sig_sq_est);
                    else
                        error('s_sq_y must be "mse" or "var"');
                    end
                    sig_sq_est = sig_sq_est * y_range^2;
                    if verbose
                        disp('bartMachine sigsq estimated...'); %only print for regression
                    end
                end
            else
                if verbose
                    disp('bartMachine using previous sigsq estimated...');
                end
            end

            %load the number of cores the user set
            num_cores = this.getset_bart_num_cores();


            java_bart_machine.setNumCores(num_cores);
            java_bart_machine.setNumTrees(num_trees);
            java_bart_machine.setNumGibbsBurnIn(num_burn_in);
            java_bart_machine.setNumGibbsTotalIterations(num_gibbs);
            java_bart_machine.setAlpha(alpha);
            java_bart_machine.setBeta(beta);
            java_bart_machine.setK(k);
            java_bart_machine.setQ(q);
            java_bart_machine.setNU(nu);
            mh_prob_steps = mh_prob_steps / sum(mh_prob_steps);
            java_bart_machine.setProbGrow(mh_prob_steps(1));
            java_bart_machine.setProbPrune(mh_prob_steps(2));
            java_bart_machine.setVerbose(verbose);
            java_bart_machine.setMemCacheForSpeed(mem_cache_for_speed);
            java_bart_machine.setFlushIndicesToSaveRAM(flush_indices_to_save_RAM);

            
            if ~isempty(seed)
                java_bart_machine.setSeed(seed);
                if num_cores > 1
                    warning("Setting the seed when using parallelization does not result in deterministic output.\nIf you need deterministic output, you must run ""set_bart_machine_num_cores(1)"" and then build the BART model with the set seed.");
                end
            end

            % now we need to set random samples
            java_bart_machine.setNormSamples(randn([3,1]));
            n_plus_hyper_nu = size(model_matrix_training_data,1) + nu;
            java_bart_machine.setGammaSamples(chi2rnd(n_plus_hyper_nu, [num_rand_samps_in_library,1]));




            if (~isempty(cov_prior_vec))
                % put in checks here for user to make sure the covariate prior vec is the correct length
                offset = length(cov_prior_vec) - (size(model_matrix_training_data, 2) - 1);
                if (offset < 0)
                    warning('covariate prior vector length = %d has to be equal to p = %d (the vector was lengthened with 1''s)', ...
                        length(cov_prior_vec), size(model_matrix_training_data, 2) - 1);
                    cov_prior_vec = [cov_prior_vec, ones(1, -offset)];
                end
                if (length(cov_prior_vec) ~= size(model_matrix_training_data, 2) - 1)
                    warning('covariate prior vector length = %d has to be equal to p = %d (the vector was shortened)', ...
                        length(cov_prior_vec), size(model_matrix_training_data, 2) - 1);
                    cov_prior_vec = cov_prior_vec(1 : (size(model_matrix_training_data, 2) - 1));
                end
                if (sum(cov_prior_vec > 0) ~= size(model_matrix_training_data, 2) - 1)
                    error('covariate prior vector has to have all its elements be positive');
                end
                java_bart_machine.setCovSplitPrior(cov_prior_vec);
            end

            if (~isempty(interaction_constraints))
                java_bart_machine.intializeInteractionConstraints(length(interaction_constraints));
                for ii = 1 : length(interaction_constraints)
                    interaction_constraint_vector = interaction_constraints{ii};
                    for b = 1 : length(interaction_constraint_vector)
                        java_bart_machine.addInteractionConstraint(interaction_constraint_vector(b), int32(interaction_constraint_vector(setdiff(1:length(interaction_constraint_vector), b))));
                    end
                end
            end

            % now load the training data into BART
            for ii = 1 : size(model_matrix_training_data, 1)
                row_as_char = string(model_matrix_training_data{ii,:});
                java_bart_machine.addTrainingDataRow(row_as_char);
            end
            java_bart_machine.finalizeTrainingData();
            if (verbose)
                fprintf('bartMachine training data finalized...\n');
            end

            % build the bart machine and let the user know what type of
            % BART this is
            if verbose
                fprintf("Now building bartMachine for %s", pred_type);
                if strcmp(pred_type, "classification")
                    fprintf(" where "" %i "" is considered the target level", y_levels(1));
                end
                disp("...")
                if ~isempty(cov_prior_vec)
                    fprintf("Covariate importance prior ON. ");
                end
                if use_missing_data
                    fprintf("Missing data feature ON. ");
                end
                if use_missing_data_dummies_as_covars
                    fprintf("Missingness used as covariates. ");
                end
                if impute_missingness_with_rf_impute
                    fprintf("Missing values imputed via rfImpute. ");
                end
            end
            java_bart_machine.Build();

            % now once it's done, let's extract things that are related to
            % diagnosing the build of the BART model
            this.java_bart_machine = java_bart_machine;
            if use_missing_data && use_missing_data_dummies_as_covars
                p_vars = p/2;
            else
                p_vars = p;
            end
            this.training_data_features = model_matrix_training_data.Properties.VariableNames(1:p_vars);
			this.training_data_features_with_missing_features = model_matrix_training_data.Properties.VariableNames(1:p); % always return this even if there's no missing features
			this.X = X;
			this.y = y;
			this.y_levels = y_levels;
			this.pred_type = pred_type;
			this.model_matrix_training_data = model_matrix_training_data;
			this.n = size(model_matrix_training_data, 1);
			this.p = p;
			this.num_cores = num_cores;
			this.num_trees = num_trees;
			this.num_burn_in = num_burn_in;
			this.num_iterations_after_burn_in = num_iterations_after_burn_in; 
			this.num_gibbs = num_gibbs;
			this.alpha = alpha;
			this.beta = beta;
			this.k = k;
			this.q = q;
			this.nu = nu;
			this.prob_rule_class = prob_rule_class;
			this.mh_prob_steps = mh_prob_steps;
			this.s_sq_y = s_sq_y;
			this.run_in_sample = run_in_sample;
			this.sig_sq_est = sig_sq_est;
			this.time_to_build = toc(t0);
			this.cov_prior_vec = cov_prior_vec;
			this.interaction_constraints = interaction_constraints;
			this.use_missing_data = use_missing_data;
			this.use_missing_data_dummies_as_covars = use_missing_data_dummies_as_covars;
			this.replace_missing_data_with_x_j_bar = replace_missing_data_with_x_j_bar;
			this.impute_missingness_with_rf_impute = impute_missingness_with_rf_impute;
			this.impute_missingness_with_x_j_bar_for_lm = impute_missingness_with_x_j_bar_for_lm;			
			this.verbose = verbose;
			this.serialize = serialize;
			this.mem_cache_for_speed = mem_cache_for_speed;
			this.flush_indices_to_save_RAM = flush_indices_to_save_RAM;
			this.debug_log = debug_log;
			this.seed = seed;
			this.num_rand_samps_in_library = num_rand_samps_in_library;


            % if the user used a cov prior vec, pass it back
            if ~null_cov_prior_vec
                this.cov_prior_vec = cov_prior_vec;
            end

            % once its done gibbs sampling, see how the training data does
            % if the user wants
            if run_in_sample
                if verbose
                    fprintf('evaluating in sample data...\n')
                end
                if strcmp(pred_type, 'regression')
                    y_hat_posterior_samples = this.java_bart_machine.getGibbsSamplesForPrediction(table2array(model_matrix_training_data), num_cores);

                    %to get y_hat.. just take straight mean of posterior samples
                    y_hat_train = mean(y_hat_posterior_samples, 2);
                    %return a bunch more stuff
                    this.y_hat_train = y_hat_train;
                    this.residuals = y_remaining - this.y_hat_train;
                    this.L1_err_train = sum(abs(this.residuals));
                    this.L2_err_train = sum(this.residuals.^2);
                    this.PseudoRsq = 1 - this.L2_err_train / sum((y_remaining - mean(y_remaining)).^2); %pseudo R^2 acc'd to our dicussion with Ed and Shane
                    this.rmse_train = sqrt(this.L2_err_train / this.n);
                elseif strcmp(pred_type, 'classification')
                    p_hat_posterior_samples = this.java_bart_machine.getGibbsSamplesForPrediction(table2array(model_matrix_training_data), num_cores);

                    %to get y_hat.. just take straight mean of posterior samples
                    p_hat_train = mean(p_hat_posterior_samples, 2);
                    y_hat_train = labels_to_y_levels(this, p_hat_train > prob_rule_class);
                    %return a bunch more stuff
                    this.p_hat_train = p_hat_train;
                    this.y_hat_train = y_hat_train;

                    %calculate confusion matrix
                    confusion_matrix = array2table(nan(3,3));
                    actual_labels = num2cell(strcat("actual ", string(y_levels)));
                    confusion_matrix.Properties.RowNames = [actual_labels(:)', {"use error"}];
                    predicted_labels = num2cell(strcat("predicted ", string(y_levels)));
                    confusion_matrix.Properties.VariableNames = [predicted_labels(:)', {"model errors"}];
                    confusion_matrix(1:2, 1:2) = confusionmat(y, y_hat_train);
                    confusion_matrix(3,1) = round(confusion_matrix(2,1) / (confusion_matrix(1,1) + confusion_matrix(2,1)), 3);
                    confusion_matrix(3,2) = round(confusion_matrix(1,2) / (confusion_matrix(1,2) + confusion_matrix(2,2)), 3);
                    confusion_matrix(1,3) = round(confusion_matrix(1,2) / (confusion_matrix(1,1) + confusion_matrix(1,2)), 3);
                    confusion_matrix(2,3) = round(confusion_matrix(2,1) / (confusion_matrix(2,1) + confusion_matrix(2,2)), 3);
                    confusion_matrix(3,3) = round((confusion_matrix(1,2) + confusion_matrix(2,1)) / sum(confusion_matrix(1:2,1:2)), 3);
                    this.confusion_matrix = confusion_matrix;
                    %obj.num_classification_errors = confusion_matrix(1, 2) + confusion_matrix(2, 1);
                    this.misclassification_error = confusion_matrix(3, 3);
                end
                if verbose
                    fprintf('done\n')
                end
            end
            if serialize
                fprintf("serializing in order to be saved for future Matlab sessions...")
                error("Serialization not supported")
                fprintf("done\n");
            end
        end
        function clone = duplicate(this, varargin)
            parser = inputParser();
            parser.addParameter("X",this.X);
            parser.addParameter("y",this.y);
            parser.addParameter("cov_prior_vec",this.cov_prior_vec);
            parser.addParameter("num_trees",this.num_trees);
            parser.addParameter("run_in_sample",false);
            parser.addParameter("covariates_to_permute",this.covariates_to_permute);
            parser.addParameter("verbose", false);
            parser.parse(varargin{:});
            clone = BartMachine(X=parser.Results.X, y=parser.Results.y, cov_prior_vec=parser.Results.cov_prior_vec, num_trees=parser.Results.num_trees, run_in_sample=parser.Results.run_in_sample, covariates_to_permute=parser.Results.covariates_to_permute, verbose=parser.Results.verbose);
        end
        function machines = bartMachineArr(this, R)
            arguments
                this
                R (1,1) {mustBeInteger} = 10
            end
            machines = [];
            machines(end+1) = this;
            for ii = 2:R
                machines(ii) = this.duplicate();
            end
        end
        function out = cov_importance_test(this, named_args)
            % Function to permute columns of X and check BART's performance
            arguments
                this {mustBeA(this,"BartMachine")}
                named_args.covariates = []
                named_args.num_permutation_samples(1,1) = 100
                named_args.plot(1,1) {mustBeLogical} = true
            end
            covariates = named_args.covariates;
            num_permutation_samples = named_args.num_permutation_samples;

            check_serialization(this); % ensure the Java object exists and
            % fire an error if not be able to handle regular expressions to
            % find the covariates

            all_covariates = this.training_data_features_with_missing_features;
            if isempty(covariates)
                title = "bartMachine omnibus test for covariates importance\n";
            elseif length(covariates) <= 3
                if isnumeric(covariates(1))
                    cov_names = strjoin(all_covariates(covariates), ', ');
                else
                    cov_names = strjoin(covariates, ', ');
                end
                title = strjoin(["bartMachine test for importance of covariate(s):", strjoin(string(cov_names),", "), "\n"]);
            else
                title = strjoin(["bartMachine test for importance of", length(covariates), "covariates\n"]);
            end
            disp(title);
            if strcmp(this.pred_type, "regression")
                observed_error_estimate = this.PseudoRsq;
            else
                observed_error_estimate = this.misclassification_error;
            end
            permutation_samples_of_error = nan(num_permutation_samples,1);
            for nsim = 1:num_permutation_samples
                fprintf(".");
                if mod(nsim,50) == 0
                    fprintf("\n");
                end
                % omnibus F-like test - just permute y (same as permuting
                % ALL the columns of X and it's faster)
                if isempty(covariates)
                    bart_machine_samp = this.duplicate(y=this.y(randperm(length(this.y))), run_in_sample=true, verbose=false); % we have to turn verbose off otherwise there would be too many outputs
                    % partial F-like test -permute the columns that we're
                    % interested in seeing if they matter
                else
                    X_samp = this.X; % copy original design matrix

                    covariates_left_to_permute = [];
                    for cov = covariates
                        if any(strcmp(cov,covariates))
                            X_samp.(cov) = X_samp.(cov)(randperm(X_samp.(cov)));
                        else
                            covariates_left_to_permute(end+1) = string(cov);
                        end
                    end
                    bart_machine_samp = this.duplicate(X=X_samp, covariates_to_permute=covariates_left_to_permute, run_in_sample=true, verbose=false); % we have to turn verbose off otherwise there would be too many outputs
                end
                % record permutation results
                if strcmp(this.pred_type, "regression")
                    permutation_samples_of_error(nsim) = bart_machine_samp.PseudoRsq;
                else
                    permutation_samples_of_error(nsim) = bart_machine_samp.misclassification_error;
                end
            end
            % compute p-value
            if strcmp(this.pred_type, "regression")
                pval = sum(observed_error_estimate< permutation_samples_of_error);
            else
                pval = sum(observed_error_estimate > permutation_samples_of_error);
            end
            pval = pval / (num_permutation_samples + 1);
            fprintf("\n");
            if named_args.plot
                title = strcat(title, "Null Samples of ");
                if strcat(this.pred_type, "regression")
                    title = strcat(title, "Pseudo-R^2's");
                else
                    title = "Misslcassification Errors";
                end
                histogram(permutation_samples_of_error, ...
                    xlim=[...
                    min(min(permutation_samples_of_error), min(0.99* observed_error_estimate)), ...
                    max(max(permutation_samples_of_error), max(1.01*observed_error_estimate))],...
                    xlabel=strcat("permutation samples\n pval = ", round(pval,3)), ...
                    brush=num_permutation_samples/10,...
                    title=title);
                hold on
                plot(observed_error_estimate, Color='blue', LineWidth=3);
            end
            fprintf("p_val = %d\n", pval);
            out = struct();
            out.permutation_samples_of_error = permutation_samples_of_error;
            out.observed_error_estimate = observed_error_estimate;
            out.pval = pval;
        end
        function out = get_var_counts_over_chain(this, type)
            % get variable counts
            arguments
                this
                type string {mustBeMember(type,["trees", "splits"])} = "splits"
            end
            this.check_serialization(); % ensure the Java object exists and fire an error if not
            C = this.java_bart_machine.getCountsForAllAttribute(type);
            C = array2table(C, "VariableNames",this.model_matrix_training_data.Properties.VariableNames{1:this.p});
            out = C;
        end
        function out = get_var_props_over_chain(this, type)
            % get variable inclusion proportions
            arguments
                this
                type string {mustBeMember(type,["trees", "splits"])}
            end
            attribute_props = this.java_bart_machine.getAttributeProps(type);
            attribute_props = array2table(attribute_props(:)', "VariableNames", this.model_matrix_training_data.Properties.VariableNames{1:this.p});
            out = attribute_props;
        end
        function out = sigsq_est(this)
            % private function called in summary
            sigsqs = this.java_bart_machine.getGibbsSamplesSigsqs();
            sigsqs_after_burnin = sigsqs((length(sigsqs) - this.num_iterations_after_burn_in):length(sigsqs));
            out = mean(sigsqs_after_burnin);
        end
        function check_serialization(this)
%             if isNull(this.java_bart_machine)
%                 error("This bartMachine object was loaded from an R image but was not serialized.\n  Please build bartMachine using the option ""serialize = TRUE"" next time.\n");
%             end
        end
        function out = predict(this, varargin)
            parser = inputParser();
            parser.addRequired("new_data", @(new_data) isa(new_data,"table"));
            parser.addParameter("type", "prob", @(type) ismember(type,["prob", "class"]));
            parser.addParameter("prob_rule_class", []);
            parser.addParameter("verbose", true);
            parser.KeepUnmatched = true;
            parser.parse(varargin{:});
            new_data = parser.Results.new_data;
            type = parser.Results.type;
            prob_rule_class = parser.Results.prob_rule_class;
            verbose = parser.Results.verbose;
            this.check_serialization(); %ensure the Java object exists and fire an error if not

            if strcmp(this.pred_type, "regression")
                out = this.get_posterior(new_data).y_hat;
            else % classification
                if strcmp(type, "prob")
                    if verbose
                        fprintf(strcat("predicting probabilities where """, this.y_levels(1), """ is considered the target level...\n"));
                    end
                    out = this.get_posterior(new_data).y_hat;
                else
                    if isempty(prob_rule_class)
                        prob_rule_class = this.prob_rule_class;
                    end
                    labels = this.get_posterior(new_data).y_hat > prob_rule_class;
                    % return whatever the raw y_levels were
                    out = this.labels_to_y_levels(labels);
                end
            end
        end
        function out = labels_to_y_levels(this, labels)
            % private function
            if labels
                out = categorical(this.y_levels(1));
            else
                out = categorical(this.y_levels(2));
            end
        end
        function out = predict_for_test_data(this, Xtest, ytest, prob_rule_class)
            arguments
                this
                Xtest {mustBeTable}
                ytest {mustBeNumericOrLogical}
                prob_rule_class = []
            end
            this.check_serialization();
            if strcmp(this.pred_type, "regression")
                ytest_hat = this.predict(Xtest);
                n = size(Xtest,1);
                L2_err = sum((ytest - ytest_hat).^2);
                out = struct();
                out.y_hat = ytest_hat;
                out.L1_err = sum(abs(ytest - ytest_hat));
                out.L2_err = L2_err;
                out.rmse = sqrt(L2_err / n);
                out.e = ytest - ytest_hat;
            else % classification
                if ~iscategorical(ytest)
                    error("ytest must be a categorical");
                end
                if ~all(ismember(categorical(ytest), this.y_levels))
                    error("New factor level not seen in training intoduced. Please remove.");
                end
                ptest_hat = this.predict(Xtest,type="prob");
                if isempty(prob_rule_class)
                    prob_rule_class = this.prob_rule_class;
                end
                ytest_labels = ptest_hat > prob_rule_class;
                ytest_hat = this.labels_to_y_levels(ytest_labels);
                confusion_matrix = nan(3);
                confusion_matrix(1:2,1:2) = confusionmat(ytest, ytest_hat);
                confusion_matrix(3, 1) = round(confusion_matrix(2, 1) / (confusion_matrix(1, 1) + confusion_matrix(2, 1)), 3);
        		confusion_matrix(3, 2) = round(confusion_matrix(1, 2) / (confusion_matrix(1, 2) + confusion_matrix(2, 2)), 3);
        		confusion_matrix(1, 3) = round(confusion_matrix(1, 2) / (confusion_matrix(1, 1) + confusion_matrix(1, 2)), 3);
        		confusion_matrix(2, 3) = round(confusion_matrix(2, 1) / (confusion_matrix(2, 1) + confusion_matrix(2, 2)), 3);
        		confusion_matrix(3, 3) = round((confusion_matrix(1, 2) + confusion_matrix(2, 1)) / sum(confusion_matrix(1 : 2, 1 : 2)), 3);
                out = struct();
                out.y_hat = yhat_test;
                out.p_hat = ptest_hat;
                out.confusion_matrix = confusion_matrix;
            end

        end
        function out = get_posterior(this, new_data)
            arguments
                this
                new_data {mustBeTable} = table()
            end
            this.check_serialization();
            if ~this.use_missing_data
                nrow_before = size(new_data,1);
                new_data = rmmissing(new_data);
                if nrow_before > size(new_data,1)
                    fprintf("%d rows omitted due to missing data. Try using the missing data feature to be able to predict on all observations.\n", nrow_before - size(new_data,1));
                end
            end

            % check for errors in data
            %
            % now process and make dummies if necessary
            new_data = BartMachine.pre_process_new_data(new_data, this);

            % check for missing data if this feature was not turned on
            if ~this.use_missing_data
                M = zeros(size(new_data));
                for ii = 1:size(new_data,1)
                    for jj = 1:size(new_data,2)
                        if isnan(new_data{ii,jj})
                            M(ii,jj) = 1;
                        end
                    end
                end
                if sum(M, "all") > 0
                    warning("missing data found in test data and bartMachine was not built with missing data feature!");
                end
            end

            y_hat_posterior_samples = this.java_bart_machine.getGibbsSamplesForPrediction(new_data{:,:}, this.getset_bart_num_cores());

            % to get y_hat just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
            y_hat = mean(y_hat_posterior_samples, 2);
            out = struct();
            out.y_hat = y_hat;
            out.X = new_data;
            out.y_hat_posterior_samples = y_hat_posterior_samples;
        end
        function out = calc_credible_intervals(this, new_data, ci_conf)
            arguments
                this,
                new_data {mustBeTable} = table()
                ci_conf (1,1) {mustBeNumeric, mustBeInRange(ci_conf,0,1)} = 0.95
            end
            this.check_serialization();
            
            % first convet the rows to the corect dummies etc
            new_data = BartMachine.pre_process_new_data(new_data, this);
            n_test = size(new_data, 1);
            ci_lower_bd = nan([1, n_test]);
            ci_upper_bd = nan([1, n_test]);
            y_hat_posterior_samples = ...% get samples
                this.java_bart_machine.getGibbsSamplesForPrediction(new_data{:,:}, this.getset_bart_num_cores());

            % to get y_hat just take straight mean of posterior samples, altenatively, we can let java do it if we want more bells and whistles
            y_hat = mean(y_hat_posterior_samples, 1);
            for ii = 1:n_test
                ci_lower_bd(ii) = quantile(sort(y_hat_posterior_samples(ii,:)), (1 - ci_conf)/2);
                ci_upper_bd(ii) = quantile(sort(y_hat_posterior_samples(ii,:)), (1 + ci_conf)/2);
            end
            % put them together and return
            out = [ci_lower_bd; ci_upper_bd];            
        end
        function out = calc_prediction_intervals(this, new_data, pi_conf, num_samples_per_data_point)
            % compute prediction intervals
            arguments
                this
                new_data {mustBeTable}
                pi_conf (1,1) {mustBeNumeric, mustBeInRange(pi_conf, 0, 1)} = 0.95
                num_samples_per_data_point (1,1) {mustBeInteger} = 1000
            end
            this.check_serialization(); % ensure the Java object exists and fire an error if not
            if strcmp(this.pred_type, "classification")
                error("Prediction intervals are not possible for classification.");
            end
            % first convet the rows to the correct dummies etc
            new_data = BartMachine.pre_process_new_data(new_data, this);
            n_test = size(new_data,1);
            pi_lower_bound = nan([n_test,1]);
            pi_upper_bound = nan([n_test,1]);
            
            y_hat_posterior_samples = ...
                this.java_bart_machine.getGibbsSamplesForPrediction(new_data{:,:}, this.getset_bart_num_cores());
            sigsqs = this.java_bart_machine.getGibbsSamplesSigsqs();

            % for each row in new_data we have to get a B x n_G matrix of draws from the normal

            all_prediction_samples = nan([n_test, num_samples_per_data_point]);
            for ii = 1:n_test
                % get all the y_hats in the posterior for this datapoint
                y_hats = y_hat_posterior_samples(ii,:);
                % make he sample of gibbs samples to pull from
                n_gs = randsample(1:this.num_iterations_after_burn_in, num_samples_per_data_point,true);
                % now make the num_samples_per_data_point draws from y_hat
                for k = 1:num_samples_per_data_point
                    y_hat_draw = y_hats(n_gs(k));
                    sigsq_draw = sigsqs(n_gs(k));
                    all_prediction_samples(ii,k) = randn(1).* sqrt(sigsq_draw) + y_hat_draw;
                end
            end

            for ii = 1:n_test
                pi_lower_bound(ii) = quantile(all_prediction_samples(ii,:), (1 - pi_conf) / 2);
                pi_upper_bound(ii) = quantile(all_prediction_samples(ii,:), (1 + pi_conf) / 2);
            end
            % put them together and return
            out = struct();
            out.interval = [pi_lower_bound, pi_upper_bound];
            out.all_prediction_samples = all_prediction_samples;
        end
    end

    methods(Static)
        function bart_num_cores = getset_bart_num_cores(n, print_out)
            arguments
                n {mustBeScalarOrEmpty, mustBePositive, mustBeInteger} = []
                print_out (1,1) logical = true
            end
            persistent bart_num_cores_;
            if isempty(bart_num_cores_)
                bart_num_cores_ = 1;
            end
            if n
                bart_num_cores_ = n;
                if print_out
                    fprintf("BartMachine now using %d cores.\n", n);
                end
            end
            bart_num_cores = bart_num_cores_;
        end
        function [bart_machine_cv, cv_stats, folds_vec] = cv(varargin)
            parser = inputParser();
            parser.KeepUnmatched = true;
            parser.addParameter("X", table());
            parser.addParameter("y", []);
            parser.addParameter("Xy", []);
            parser.addParameter("num_trees_cvs", [50,200]);
            parser.addParameter("k_cvs", [2,3,5]);
            parser.addParameter("nu_q_cvs", []);
            parser.addParameter("k_folds", 5);
            parser.addParameter("folds_vec", []);
            parser.addParameter("verbose", false);
            parser.parse(varargin{:});
            X = parser.Results.X;
            y = parser.Results.y;
            Xy = parser.Results.X;
            num_trees_cvs = parser.Results.num_trees_cvs;
            k_cvs = parser.Results.k_cvs;
            nu_q_cvs = parser.Results.nu_q_cvs;
            k_folds = parser.Results.k_folds;
            folds_vec = parser.Results.folds_vec;
            verbose = parser.Results.verbose;
            if (isempty(X) && isempty(Xy)) || (isempty(y) && isempty(Xy))
                error("You need to give bartMachine a training set either by specifying X and y or by specifying a matrix Xy which contains the response named ""y.""\n");
            elseif ~isempty(X) && ~isempty(y) && ~isempty(Xy)
                error("You cannot specify both X,y and Xy simultaneously.")
            elseif ~isempty(X) && isempty(y)
                y = Xy.y;
                Xy.y = [];
                X = Xy;
            end
            if ~isempty(folds_vec) && ~all(folds_vec == round(folds_vec))
                error("folds_vec must be an a vector of integers specifying the indexes of each folds.")
            end
            y_levels = unique(y);
            if isnumeric(y) % if y is numeric, then it's a regression problem
                pred_type = "regression";
            elseif iscategory(y) && (length(y_levels) == 2) % if y is a categorical and and binary, then it's a classification problem
                pred_type = "classification"; 
            else % otherwise throw an error
                error("Your response must be either numeric, an integer or a categorical with two levels.");
            end

            if strcmp(pred_type, "classification")
                if ~isempty(nu_q_cvs)
                    error("For classification, ""nu_q_cvs"" must be set to NULL (the default).");
                end
                nu_q_cvs = {[3, 0.9]} % ensure we only do this once, the 3 and the 0.9 don't actually matter, they just need to be valid numbers for the hyperparameters
            else
                % i.e. regression...
                if isempty(nu_q_cvs)
                    nu_q_cvs = {[3,0.9], [3,0.99], [10,0.75]};
                end
            end

            min_rmse_num_tree = [];
        	min_rmse_k = [];
        	min_rmse_nu_q = [];
        	min_oos_rmse = Inf;
        	min_oos_misclassification_error = inf;

            cv_stats = array2table(nan(length(k_cvs) * length(nu_q_cvs) * length(num_trees_cvs), 6));
            cv_stats.Properties.VariableNames = ["k", "nu", "q", "num_trees", "oos_error", "% diff with lowest"];

            % set up k folds
            if isempty(folds_vec) % if folds were not pre-set:
                n = size(X,1);
                if k_folds == inf % leave-one-out
                    k_folds = n;
                end
                if k_folds <= 1 || k_folds > n
                    error("The number of folds must be at least 2 and less than or equal to n, use ""inf"" for leave one out");
                end
                temp = randn(n);
                folds_vec = discretize(temp, quantile(temp, linspace(0, 1, k_folds + 1)), 'IncludedEdge', 'left');
            else
                k_folds = length(unique(folds_vec)); % otherwise we know the folds, so just get k
            end

            % cross-validate
            run_counter = 1;
            for k = k_cvs
                for nu_q_idx = 1:length(nu_q_cvs)
                    nu_q = nu_q_cvs{nu_q_idx};
                    for num_trees = num_trees_cvs
                        if strcmp(pred_type, "regression")
                            fprintf(" BartMachine CV try: k: %i, nu, q: %f, %f, m: %i\n", k, nu_q(1), nu_q(2), num_trees);
                        else
                            fprintf(" BartMachine CV try: k: %i, m: %i\n", k, num_trees);
                        end
                        vararg_cells = namedargs2cell(parser.Unmatched);
                        k_fold_results = BartMachine.k_fold_cv('X', X, 'y', y, ...
                        'k_folds', k_folds,...
    					'folds_vec', folds_vec,... % will hold the cv folds constant
    					...
                        'num_trees', num_trees,...
    					'k', k,...
    					'nu', nu_q(1),...
    					'q', nu_q(2),...
    					'verbose', verbose,...
    					vararg_cells{:});
                        if strcmp(pred_type, "regression") && k_fold_results.rmse < min_oos_rmse
                            min_oos_rmse = k_fold_results.rmse;
                            min_rmse_k = k;
                            min_rmse_nu_q = nu_q;
                            min_rmse_num_tree = num_trees;
                            err = k_fold_results.rmse;
                        elseif strcmp(pred_type,"classification") & k_fold_results.misclassification_error < min_oos_misclassification_error
                            min_oos_misclassification_error = k_fold_results.misclassification_error;
                            min_rmse_k = k;
                            min_rmse_nu_q = nu_q;
                            min_rmse_num_tree = num_trees;
                            err = k_fold_results.misclassification_error;
                        end
                        
                        cv_stats(run_counter, 1:5) = {k, nu_q(1), nu_q(2), num_trees, err};
                        run_counter = run_counter + 1;
                    end
                end
            end
            if strcmp(pred_type,"regression")
                fprintf(" BartMachine CV win: k: %i, nu, q: %f, %f, m: %i\n", min_rmse_k, min_rmse_nu_q(1), min_rmse_nu_q(2), min_rmse_num_tree);
            else
                fprintf(" BartMachine CV win: k: %i, m: %i\n", min_rmse_k, min_rmse_num_tree);
            end
            % now that we've found the best settings, return that bart
            % machine. It would be faster to have kept this around, but
            % doing it this way saves RAM for speed.
            vararg_cells = namedargs2cell(parser.Unmatched);
            bart_machine_cv = BartMachine('X', X, 'y',y, 'num_trees', min_rmse_num_tree, 'k', min_rmse_k, 'nu', min_rmse_nu_q(1), 'q', min_rmse_nu_q(2), vararg_cells{:});
            % give the user some cv_stats ordered by the best (ie lowest)
            % oosrmse
            [~,perm_order] = sort(cv_stats.oos_error);
            cv_stats = cv_stats(perm_order,:);
            cv_stats{:,6} = (cv_stats{:,5} - cv_stats{1,5}) / cv_stats{1,5} * 100;
        end
        function X_with_missing = imputeMatrixByXbarjContinuousOrModalForBinary(X_with_missing, X_for_calculating_avgs)
            arguments
                X_with_missing table
                X_for_calculating_avgs table
            end
            for ii = 1:size(X_with_missing,1)
                for jj = 1:size(X_with_missing,2)
                    if isnan(X_with_missing{ii,jj})
                        % mode for factors, otherwise average
                        if iscategorical(X_with_missing{:,jj})
                            X_with_missing{ii,jj} = mode(X_for_calculating_avgs{:,jj});
                        else
                            X_with_missing{ii,jj} = mean(X_for_calculating_avgs{:,jj},'omitnan');
                        end
                    end
                end
            end
            % now we have to go through and drop columns that are all NaN's
            % if need be
            bad_cols = false(1,size(X_with_missing,2));
            for jj = 1:size(X_with_missing, 2)
                if sum(isnan(X_with_missing{:,jj}) == size(X_with_missing,1))
                    bad_cols(jj) = true;
                end
            end
            X_with_missing(:,bad_cols) = [];
        end
        function prediction = predict_bartMachineArr(bart_machines, new_data, varargin)
            R = length(bart_machines);
            n_star = size(new_data,1);
            predicts = nan(n_star,R);
            for r = 1:R
                predicts(:,r) = predict(bart_machines(r),new_data,varargin{:});
            end
            prediction = mean(predicts,2);
        end
        % performs out-of-sample error estimation for a BART model
        function out = k_fold_cv(varargin)
            parser = inputParser();
            parser.addRequired("X",istable);
            parser.addRequired("y",@(y) isnumeric(y) || iscategorical(y));
            parser.addParameter("k_folds",5);
            parser.addParameter("folds_vec", [], @(v) all(round(v) == v));
            parser.addParameter("verbose",false);
            parser.KeepUnmatched = true;
            parser.parse(varargin{:});
            % we cannot afford the time sink of serialization during the
            % grid search, so shut it of manually
            parser.Results.serialize = false;
            X = parser.Results.X;
            y = parser.Results.y;
            k_folds = parser.Results.k_folds;
            folds_vec = parser.Results.folds_vec;
            verbose = parser.Results.verbose;
            
            if isnumeric(y)
                pred_type = "regression";
            elseif iscategorical(y)
                pred_type = "classification";
                y_levels = unique(y);
            end

            n = size(X, 1);
            Xpreprocess = BartMachine.pre_process_training_data(X).data;

            p = size(Xpreprocess, 2);

            % set up k folds
            if isempty(folds_vec) % if folds were not pre-set:
                if k_folds == Inf % leave-one-out
                    k_folds = n;
                end

                if k_folds <= 1 || k_folds > n
                    error("The number of folds must be at least 2 and less than or equal to n, use ""Inf"" for leave one out");
                end

                temp = randn(n, 1);

                folds_vec = discretize(temp, quantile(temp, 0:1/k_folds:1), 'IncludedEdge', 'left');
            else
                k_folds = numel(unique(folds_vec)); % otherwise we know the folds, so just get k
            end

            if strcmp(pred_type, "regression")
                L1_err = 0;
                L2_err = 0;
                yhat_cv = zeros(n, 1); % store cv
            else
                phat_cv = zeros(n, 1);
                yhat_cv = categorical(n, y_levels);
                col_names = [strcat("predicted ", string(y_levels)), "model errors"];
                row_names = [strcat("actual ", string(y_levels)), "actual errors"];
                confusion_matrix = array2table(zeros(3), 'VariableNames', col_names, 'RowNames', row_names);
            end
            Xy = array2table(Xpreprocess.data);
            Xy.y = y;
            for k = 1:k_folds
                fprintf('.');
                train_idx = folds_vec ~= k;
                test_idx = setdiff(1:n, train_idx);
                test_data_k = Xy(test_idx, :);
                training_data_k = Xy(train_idx, :);

                % build bart object
                vararg_cells = namedargs2cell(parser.Unmatched);
                bart_machine_cv = BartMachine("X", training_data_k(:,1:p), "y", training_data_k.y, "run_in_sample", false, "verbose", verbose, vararg_cells{:});
                predict_obj = bart_predict_for_test_data(bart_machine_cv, test_data_k(:, 1:p), test_data_k.y);

                % tabulate errors
                if strcmp(pred_type, "regression")
                    L1_err = L1_err + predict_obj.L1_err;
                    L2_err = L2_err + predict_obj.L2_err;
                    yhat_cv(test_idx) = predict_obj.y_hat;
                else
                    phat_cv(test_idx) = predict_obj.p_hat;
                    yhat_cv(test_idx) = predict_obj.y_hat;
                    tab = confusionmat(categorical(test_data_k(:, p+1), y_levels), categorical(predict_obj.y_hat, y_levels));
                    confusion_matrix(1:2, 1:2) = confusion_matrix(1:2, 1:2) + tab;
                end
            end
            fprintf("\n");
            out = struct();
            if strcmp(pred_type, "regression")
                out.y_hat = yhat_cv;
                out.L1_err=L1_err;
                out.L2_err = L2_err;
                out.rmse = sqrt(L2_err /n);
                out.PseudoRsq = 1 - Ls_err / sum((y - mean(y))^2);
                out.folds = folds_vec;
            else
                % calculate the rest of the confusion matrix and return it plus the
                % errors
                confusion_matrix(3, 1) = round(confusion_matrix(2, 1) / (confusion_matrix(1, 1) + confusion_matrix(2, 1)), 3);
        		confusion_matrix(3, 2) = round(confusion_matrix(1, 2) / (confusion_matrix(1, 2) + confusion_matrix(2, 2)), 3);
        		confusion_matrix(1, 3) = round(confusion_matrix(1, 2) / (confusion_matrix(1, 1) + confusion_matrix(1, 2)), 3);
        		confusion_matrix(2, 3) = round(confusion_matrix(2, 1) / (confusion_matrix(2, 1) + confusion_matrix(2, 2)), 3);
        		confusion_matrix(3, 3) = round((confusion_matrix(1, 2) + confusion_matrix(2, 1)) / sum(confusion_matrix(1 : 2, 1 : 2)), 3);
                out.y_hat = y_hat_cv;
                out.p_hat = p_hat_cv;
                out.confusion_matrix.confusion_matrix;
                out.misclassification_error = confusion_matrix(3,3);
                out.folds = folds_vec;
            end
        end
        function data = dummify_data(data)
            data = array2table(BartMachine.pre_process_training_data(data).data);
        end
        function out = pre_process_training_data(data, use_missing_data_dummies_as_covars, imputations)
            arguments
                data {mustBeTable}
                use_missing_data_dummies_as_covars (1,1) {mustBeLogical} = false
                imputations = []
            end
            % first convert characters to factors
            factors = [];
            for col = 1:size(data,2)
                if isstring(data{:,col})
                    data{:,col} = categorical(data{:,col});
                    factors(end+1) = data.Properties.VariableNames{col};
                end
                if ischar(data{:,col})
                    data{:,col} = categorical(cellstr(data{:,col}));
                    factors(end+1) = data.Properties.VariableNames{col};
                end
            end
            factor_lengths = zeros(length(factors),1);
            for ii=1:length(factors)
                fac = factors(ii);
                % first create the dummies to be appended for this factor
                dummied = onehotencode(data.(fac),2);
                % ensure they are named appropriately
                col_names = [strcat(fac, "_", string(unique(data.(fac))))];
                % append them to the data
                data = [data, table(dummied, 'VariableNames', col_names)];
                % delete the factor covariates from the design matrix
                data.(fac) = [];
                % record the length of this factor
                factor_lengths(ii) = size(dummied,2);
            end
            if use_missing_data_dummies_as_covars
                % now take care of missing data - add each column as a
                % missingness dummy
                predictor_columns_with_missingness = sum(isnan(table2array(data)), 2);
                % only do something if there are preictors with missingness
                if ~isempty(predictor_columns_with_missingness)
                    M = zeros(length(predictor_columns_with_missingness), size(data,1));
                    for ii = 1:size(data,1)
                        for jj = 1:length(predictor_columns_with_missingness)
                            if ismissing(data{ii, predictor_columns_with_missingness(jj)})
                                M(ii,jj) = 1;
                            end
                        end
                    end
                    col_names = strcat("M_",data.Properties.VariableNames{predictor_columns_with_missingness});
                    % now we may want to add imputations before the
                    % missingness dummies
                    if ~isempty(imputations)
                        data = [data, imputations];
                    end
                    % append the missing dummy columns to data as if
                    % they're real attributes themselves
                    data = [data, M];
                end
            elseif ~isempty(imputations)
                data = [data, imputations];
            end
            out = struct();
            out.data = data;
            out.factor_lengths = factor_lengths;
        end
        function out = pre_process_new_data(new_data, bart_machine)
            arguments
                new_data {mustBeTable}
                bart_machine BartMachine
            end
            n = size(new_data,1);
            imputations = [];
            if bart_machine.impute_missingness_with_rf_impute
                % we have to impute with missForest since we don't have y's
                % for the test data we want to predict
                % TODO: R uses the MissForest package, which is not
                % available in Matlab. 
                % https://de.mathworks.com/help/finance/impute-missing-data-using-random-forest.html
                error('Random forest imputation not yet implemented');
            end
            % preprocess the new data with the training data to ensure
            % proper dummies
            new_data_and_training_data = [new_data;bart_machine.X];
            % kill all factors again
            predictors_which_are_factors = [];
            for ii = 1:size(new_data_and_training_data,2)
                col_name = new_data_and_training_data.Properties.VariableNames{ii}; 
                if iscategorical(new_data_and_training_data.(col_name))
                    new_data_and_training_data.(col_name) = categorical(new_data_and_training_data.(col_name));
                    predictors_which_are_factors(end+1) = string(col_name);
                end
            end
            new_data = BartMachine.pre_process_training_data(new_data_and_training_data, bart_machine.use_missing_data_dummies_as_covars, imputations).data;
            if bart_machine.use_missing_data
                training_data_features = bart_machine.training_data_features_with_missing_features;
            else
                training_data_features = bart_machine.training_data_features;
            end

            % The new data features has to be a superset of the training
            % data features, so pair it down even more
            new_data_features_before = new_data.Properties.VariableNames;
            new_data = new_data(1:n, training_data_features);
            differences = setdiff(new_data_features_before, training_data_features);
            if ~isempty(differences)
                warning_msg = "The following features were found in records for prediction which were not found in the original training data:\n    ";
                for difference = differences
                    warning_msg = strcat(warning_msg, difference, ", ");
                end
                warning_msg = strcat(warning_msg, "\n  These features will be ignored during prediction.");
                warning(warning_msg);
            end
            new_data_features = new_data.Properties.VariableNames;
            if ~all(strcmp(new_data_features,training_data_features))
                warning("Are you sure you have the same feature names in the new record(s) as the training data?");
            end
            % iterate through and see
            for jj = 1:length(training_data_features)
                training_data_feature = training_data_features(jj);
                new_data_feature = new_data_features(jj);
                if ~strcmp(training_data_feature,new_data_feature)
                    % create the new col of zeros, give it the same name as in the training set
                    new_col = array2table(zeros(n,1), VariableNames={training_data_feature});
                    % wedge it into the data set
                    temp_new_data = [new_data(:,1:(jj-1)), new_col];
                    % tack on the rest of the stuff
                    if size(new_data,2) >= jj
                        rhs = new_data(:, jj:size(new_data,2));
                        if isnumeric(rhs)
                        end
                        temp_new_data = [temp_new_data, rhs];
                    end
                    new_data = temp_new_data;
                    % update list
                    new_data_features = new_data.Properties.VariableNames;
                end
            end
            out = new_data;
        end
        function out = linearity_test(varargin)
            parser = inputParser();
            parser.KeepUnmatched = true;
            parser.addParameter("lin_mod", []);
            parser.addParameter("X", table());
            parser.addParameter("y", []);
            parser.addParameter("num_permutation_samples", 100);
            parser.addParameter("plot", true);
            parser.parse(varargin{:});
            lin_mod = parser.Results.lin_mod;
            X = parser.Results.X;
            y = parser.Results.y;
            num_permutation_samples = parser.Results.num_permutation_samples;
            plot_ = parser.Results.plot;
            if isempty(lin_mod)
                lin_mod = fitlm(table2array(X),y);
            end
            y_hat = predict(lin_mod, X);
            vararg_cells = namedargs2cell(parser.Unmatched);
            bart_mod = BartMachine(X, y - y_hat, vararg_cells{:});
            out = cov_importance_test(bart_mod, num_permutation_samples= num_permutation_samples, plot = plot_);
        end

    end
end

