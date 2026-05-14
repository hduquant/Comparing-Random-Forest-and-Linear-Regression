#setting search path for R libraries to custom directory, where our R custom 
# - directory path for Hoff is prepended to the current library paths of .libPaths()
.libPaths(c("~/R/x86_64-pc-linux-gnu-library/4.2.2", .libPaths()))

# Now in poper dir, randomForest package being loaded and supressing the startup messages
# - so that our logs are cleaner when check output 
suppressPackageStartupMessages({ library(randomForest) })

# our command line arguments passed to the script are below
# - actually only want the stuff after our script name so trailing Only
args <- commandArgs(trailingOnly = TRUE)

# our needed arguments to run:<N> <p> <Model> <Method> <ntrees> <reps_per_job> 
# [block] optional while others required, but blocking was good strat for seed set consistency cross-models
# when running Rscript rf_vs_reg_worker_5050.R 30 6 A rf 500 20 1:
# command-line args coming after our script name are the ones being read below...
if (length(args) < 6) {
  stop("Usage: Rscript rf_vs_reg_worker_5050.R <N> <p> <Model> <Method> <ntrees> <reps_per_job> [block]")
}
# therefore, if we are missing any of the 6 req, will stop the script and give us the above error mess
# without this, we know script not working as intended & would have issues downstream (i.e, emitted args[6])

# Now, the below command line args comin in as strings that we will convert to integers:
N            <- as.integer(args[1]) # training samp size that is also our testing n, given 50/50 split
p            <- as.integer(args[2]) # Feature size, so X1... Xp
model        <- args[3] # Coded as string whether "A", "B", ect.
method       <- args[4]  # "reg" and "rf"      
ntrees       <- as.integer(args[5])  # no. trees if rf, but ignored if reg 
reps_per_job <- as.integer(args[6])  # replications run in this particular job 
block        <- if (length(args) >= 7) as.integer(args[7]) else 1L # If our length is 7 or more, just grabbing
# -th as block, otherwise 1L to just give it integer of 1, so our block is just set to 1

# On sun grid envir, scheduler setting our environ variables s.t. sys.getenv gives us the envir var as a sring
# Our SGE task id is just which task number this job is inb the array
# if we are just running this locally, we won't have these environ var, so we can set to "local" for this identification
task_id <- Sys.getenv("SGE_TASK_ID"); if (task_id == "") task_id <- "local"
# Our job id is the identifier for overall submission; one you kill jobs with
job_id  <- Sys.getenv("JOB_ID");      if (job_id  == "") job_id  <- "local"

# to ensure our models are only the ones intended, we can erect some guardrails here:
# The models we okay are only the below set, and if there is a typo, where the model is not allowed, then 
# - we will use the stop argument to print that the model with error message and stop script
ok_models <- c("A","B","C","F","G2","H2","K")
if (!model %in% ok_models) stop("Model not in allowed set: ", model)

# Move on to production of our constants: b0 and beta that are used to generate our Y
b0   <- 1.0 # just the intercept of our dgm, chosen as 1 for interpretability 
beta <- 0.2 # our base coefficient, being used for both our primary linear effects and the extra parts
log_safe <- function(x, eps = 1e-8) log(pmax(x, eps)) # flooring our threshold of 1e-8, use parallel maximization
# - here to be able to look at every # in x and determine if x > eps; guards against 0, neeg #, and most extreme near-0 cases rounded down to 0;
# instead going to set it to veru large neg # as ln(eps) should be about -18 or so, preventing -inf for rounded to 0 and Nan for neg cases

# Our make_x fn. here is going to generate pred matrix X of n x p dim, feeding n, p and model as arg
# And for G2, we are using chi-sq predictors that are positive and skewed, 2 df instead of normal
make_X <- function(n, p, mdl) {
  if (mdl == "G2") matrix(rchisq(n*p, df=2), ncol=p) else matrix(rnorm(n*p), ncol=p)
  # If we are working with chisq in G2, then want to draw n*p chi sq (on 2 df) random numbers with rchisq fn.
  # Then we want to use matrix fn with these drawn n* p drawn chi sq random #'s into a matrix column-wise with p col
  # if we are not working with G2 and chisq distr. then we will have rnorm(n*p) to draw n*p standard normal random #'s
  # Then we want to use matrix fn with these drawn values to put them into matrix column-wise with p coluns 
}

# extra part that is giving us the nonlin/ interact aspect of model tacked upon the linear backbone
# extra part fn. taking in X and our model, then returning a vector length n, where we are building one extra val per row 
extra_part <- function(X, mdl) {
  # nrow as the no. obs, w ncol as no. predictors
  # numeric(n) initializing vector of 0's length n; out starting at 0, then adding contributions depending on model 
  n <- nrow(X); p <- ncol(X); out <- numeric(n)
  # If our model is B or C, then we add some sort of 2 and three way interactions 
  if (mdl %in% c("B","C")) {
    # floor(p/5), so lower bound of this cutoff for two way interaction terms; we try n2 as no. of 2 way int terms
    # only run loop if n2 > 0, as otherwise would be 1:n2 wb 1:0
    n2 <- floor(p/5); if (n2 > 0) for (j in 1:n2) if (j+1 <= p) out <- out + beta * (X[,j]*X[,j+1])
    # floor(p/4), so lower bound of this cutoff for three way interaction terms; we try n3 as no. of 3 way int terms
    # only run loop if n3 > 0, as otherwise would be 1:n3 wb 1:0
    # for j = 1...n2, we add beta * Xj * X(j+1), with if (j+1 <= p) as a safety check so that we don't index past col p to smthn doesn't exist
    n3 <- floor(p/4); if (n3 > 0) for (k in 1:n3) if (k+2 <= p) out <- out + beta * (X[,k]*X[,k+1]*X[,k+2])
    #no. 3 way int. terms attempted above
    # for k = 1...n3, add beta * Xk * X(k+1) * X(k+2), with if (k+2 <= p) as our safety check so that we don't index past col p to smthn non=exist
  }
  # If model is C, we will add poly terms where for j = 1...floor(p/3): adding beta * Xj^(j+1)
  # For this, our exp increases with j, so X1^2, X2^3... so on...
  if (mdl == "C") { np <- floor(p/3); if (np > 0) for (j in 1:np) out <- out + beta * (X[,j]^(j+1)) }
  # If our model is F, we add exponential terms where for j =1... floor(p/3): adding beta * exp(Xj)
  if (mdl == "F") { k <- floor(p/3); if (k > 0) for (j in 1:k) out <- out + beta * exp(X[,j]) }
  # If our model is G2, we add log terms where for j = 1...floor(p/3): add beta * log(Xj); since G2 using chisq pred, they're nonneg
  # These values could still be very close to 0 and ultimately get inf reading, so our log_safe object prevents -Inf occurence
  if (mdl == "G2"){ k <- floor(p/3); if (k > 0) for (j in 1:k) out <- out + beta * log_safe(X[,j], 1e-8) }
  #If our model is H2, we are adding cosine nonlinearities for X1 and X2
  # - with different weights, where we only do this if p large enough to have these col
  if (mdl == "H2"){ if (p >= 1) out <- out + 1.1*cos(X[,1]); if (p >= 2) out <- out + 1.2*cos(X[,2]) }
  # If model is K, we add cosnine terms with shift where for j = 1... floor(p/3): adding (j/10) * beta * cos(Xj + pi/j)
  # Here, our coeff grow with j thanks to the (j/10) element included
  if (mdl == "K") { k <- floor(p/3); if (k > 0) for (j in 1:k) out <- out + (j/10) * beta * cos(X[,j] + pi/j) }
  # importantly, in the end we are returning the vector of extra contributions that extend beyond our linear backbone
  # This is the nonlinear aspect we are adding...
  out
}
# Next, we will generate a full dataset wiht columns Y and X1...Xp, where Y is gen according to:
# Y = b0 + beta * sum(X row) + extra_part(X) + random noise 
# To accomplish this, we are feeding n, p, and our model
make_data <- function(n, p, mdl) {
  # Generate our predictor matrix, where we will name the columns X1...Xp as strings
  X <- make_X(n, p, mdl); colnames(X) <- paste0("X", 1:p)
  # rowSums(X) to give us a vector of length n that represents the sum of each row's predictors
  # rnorm(n) adds ind. N(0,1) noise; using rowSums as beta's are same, could've gone %*% operator if different betas to get (nx1) matr
  # keep in mind, if want a vec of betas, then would have to change the beta aspect in the extra_part fn., so would want a seaparte constant for interactions.. 
  Y <- b0 + beta * rowSums(X) + extra_part(X, mdl) + rnorm(n)
  # Data frame where we will make Y before all X columns, and we don't want our column names altered
  data.frame(Y=Y, X, check.names=FALSE)
}

# For fit eval fn. we want to subset train/test to keep only Y and X1...Xp, as messed up on this previously, so making more explicit
# Check for numeric validitiy w/ no NA/Inf; fit lm() or randomForest(); predict on test, returning MSE
# seed_fit_rf for RF internal randomness; set to default val of int version of NA
# - working w frameowrk for res df expecting SeedFit col to be int & doesn't crash under lr
fit_eval <- function(train, test, method, p, ntrees, seed_fit_rf = NA_integer_) {
  # First, defining the col want to keep to ensure only using the first p predictors; not accidentally including extra col if existent
  keep <- c("Y", paste0("X", 1:p))
  # Subset training and testing, with drop = FALSE to make sure that our result stays a df; don't want R to switch data type under case of small p or col missing
  # We are expecting a df w/ rows and cols here for lm, randomForest, and predict...
  tr <- train[, keep, drop=FALSE]
  te <- test[,  keep, drop=FALSE]
  
  # Safety check here to check whether any NA, NaN, Inf in train or test, as we cannot then fit and predict reliably
  # as.matrix(tr) putting entire training and testing data frames into matrices to enables is.finite to work (tr and te r data frames which are lists of cols)
  # So, we want to be able to check all of our cells at once; checking to see whether a number is a normal # instaed of NA, NaN or Inf, -Inf that will return false for the !is.finite
  # then any true values, so if just a cell is Inf or NA or NaN, we will return true 
  # if there is any bad data, then we get a result taht matches the format of our successful runs but with empty and error vals
  # We coordinate the specific type of NA ensures that the column in our results table stays way we want it .. (i.e, NA_real_ helping us stay numeric for this col in results table)
  # Ok = 0 as flag for run, so later can subset ok = 1 for good runs 
  # coef_ok, rank_ok, ect. as NA_integer_ to act as a placeholder that aligns with col types, and we haven't even run reg yet
  # Still logging the settings for trees here as we want to know what settings may have caused a failure, (i.e, memory impact on crazy high # trees)
  if (any(!is.finite(as.matrix(tr))) || any(!is.finite(as.matrix(te)))) {
    return(list(mse=NA_real_, ok=0L, coef_ok=NA_integer_, coef_len=NA_integer_,
                rank_ok=NA_integer_, trees=ifelse(method=="rf", ntrees, NA_integer_),
                mtry=ifelse(method=="rf", max(1L,floor(p/2)), NA_integer_),
                seed_fit=ifelse(method=="rf", seed_fit_rf, NA_integer_)))
  }
  
  
  if (method == "reg") {
    # fit a lr using all pred in tr beside Y; Y ~ . to include all col bc already subset to just X1...Xp
    # Gonna use try(..., silent = TRUE) to prevent any crashing of aentire script on an error, instead working with "try-error" object 
    f  <- try(lm(Y ~ ., data = tr), silent = TRUE)
    # If our lm() has failed, we see a "try-error", then we want to return the failure information, where we are explicitly saying the coefs are not okay
    if (inherits(f, "try-error"))
      return(list(mse=NA_real_, ok=0L, coef_ok=0L, coef_len=NA_integer_,
                  rank_ok=NA_integer_, trees=NA_integer_, mtry=NA_integer_, seed_fit=NA_integer_))
    # maodel.matrix(f) as the design matrix used by lm, which includes intercept col and col for each of the pred
    mm <- model.matrix(f)
    #coef(f) gives us all of our estimated coefficients: Intercept and slopes for X1...Xp
    co <- coef(f)
    # With this information, we can check to see if our coef are able to be used...
    # There should be exactly p+1 values (intercept and p slopes); all should be finite; and, no NA anywhere
    coef_ok <- as.integer(length(co) == (p+1) && all(is.finite(co)) && !any(is.na(co)))
    # Now, want a rank check to see if okay if model matrix was fulll rank by checking if the # of unique pieces of info the model found & used was same as what we expect in our model matrix: ncol(mm)
    rank_ok <- as.integer(f$rank == ncol(mm))
    # predict on test set
    pr <- try(predict(f, newdata = te), silent = TRUE)# If prediction failed, our predictions are not finite OR coefficients are invalid, then we log a failure
    if (inherits(pr, "try-error") || any(!is.finite(pr)) || coef_ok==0L) {
      return(list(mse=NA_real_, ok=0L, coef_ok=coef_ok, coef_len=length(co),
                  rank_ok=rank_ok, trees=NA_integer_, mtry=NA_integer_, seed_fit=NA_integer_))
    }
    # Compute MSE on test as the mean of squared residuals between our true Y and pred Y
    mse <- mean((te$Y - pr)^2)
    # Return a list of results/ diagnositics for the successful run
    return(list(mse=mse, ok=1L, coef_ok=coef_ok, coef_len=length(co),
                rank_ok=rank_ok, trees=NA_integer_, mtry=NA_integer_, seed_fit=NA_integer_))
    # If we are not uisng reg, then random forest!
  } else {
    # ensure mtry is at least 1, but we are working with floor(p/2)
    mtry_val <- max(1L, floor(p/2))
    # With the bagging and random feature selection, want to set seet for RF internal randomness, so we do that...
    if (is.finite(seed_fit_rf)) set.seed(seed_fit_rf)
    # fit a random forest whre y ~ . using all pred in tr that is already subset, with ntree and mtry as hyperparam
    f  <- try(randomForest(Y ~ ., data = tr, ntree = ntrees, mtry = mtry_val), silent = TRUE)
    # If RF fit failed, return the failure
    if (inherits(f, "try-error"))
      return(list(mse=NA_real_, ok=0L, coef_ok=NA_integer_, coef_len=NA_integer_,
                  rank_ok=NA_integer_, trees=ntrees, mtry=mtry_val, seed_fit=seed_fit_rf))
    #predict on the test set here
    pr <- try(predict(f, newdata = te), silent = TRUE)
    # If our prediction failed or was non finite, then we log it as a failure 
    if (inherits(pr, "try-error") || any(!is.finite(pr))) {
      return(list(mse=NA_real_, ok=0L, coef_ok=NA_integer_, coef_len=NA_integer_,
                  rank_ok=NA_integer_, trees=ntrees, mtry=mtry_val, seed_fit=seed_fit_rf))
    }
    # Compute our test MSE 
    mse <- mean((te$Y - pr)^2)
    # Return list containing the elements of our successful trial (remember coef_ok,coef_len, and rank_ok are not applicable to RF)
    return(list(mse=mse, ok=1L, coef_ok=NA_integer_, coef_len=NA_integer_,
                rank_ok=NA_integer_, trees=ntrees, mtry=mtry_val, seed_fit=seed_fit_rf))
  }
}

# in preparation for main loop, we prepare an index for model id that will just tell us the poistion of the model in ok_models (i.e, model="B", then model_id = 2)
# Want to use model_id in the seeding formulas so that each model will have its own seeding stream
model_id <- match(model, ok_models)
# Initializing an empty dataframe to store our results here, where we'll add one addt'l row per rep
res <- data.frame()
# want to count the number of reps that were failed where our MSE is NA or ok=0
skipped <- 0L
# The main simulation loop:
# Deterministic seeds for training data gen, test data gen, RF fitting when applicable
# in each, we use the formaulas to have seeds depend on the condituib parameters and the replication index
# Different conditions give us different seeds so that they don't reuse the same random draws
# Different blocks give us different seeds so that array tasks don't repeat one another
# Different r give us different replications within the block
for (r in 1:reps_per_job) {
  seed_tr  <- as.integer(1009*N + 9173*p + 53*model_id + 7919*block + r)
  seed_te  <- as.integer(2003*N + 9341*p + 59*model_id + 8011*block + r)
  # Just RF needs a fit seed bc of the model's randomness in its fitting procedure
  seed_fit <- if (method == "rf") as.integer(3001*N + 9397*p + 61*model_id + 8081*block + r) else NA_integer_
  # Generate training data using seed_tr
  set.seed(seed_tr); train <- make_data(N, p, model)
  # Generate test data using seed_te
  # having these different set seeds allows us to know that we have kept the test independent from train 
  set.seed(seed_te); test  <- make_data(N, p, model)
  # fit the model and compute MSE 
  out <- fit_eval(train, test, method, p, ntrees, seed_fit_rf = seed_fit)
  # If it failed where MSE is NA or ok==0, then we count it
  if (is.na(out$mse) || out$ok == 0L) skipped <- skipped + 1L
  # Then, we want to append a row of results, so we use rbind() fn. to glue a nrew row onto res.
  res <- rbind(res, data.frame(
    N_train = N, N_test = N, p = p, Model = model, Method = method,
    Block = block, Rep = r,
    #grabbing some hyperparam for RF (NA under reg)
    Trees = out$trees, mtry = out$mtry,
    # getting our main outcome and whether converged
    MSE = out$mse, Converged = out$ok,
    # Get some reg diagnostics (NA for RF)
    CoefOK = out$coef_ok, CoefLen = out$coef_len, RankOK = out$rank_ok,
    # Grabbing seeds so can reproduce any replication perfectly
    SeedTrain = seed_tr, SeedTest = seed_te, SeedFit = out$seed_fit,
    # Save the cluster IDs in case needed to trace 
    TaskID = task_id, JobID = job_id,
    # Make sure strings are not factors
    stringsAsFactors = FALSE
  ))
}
# Decide on output directory where if the RESULTS_DIR exists, we use it, otherwise default to results, like local runs and the CWD
out_dir <- Sys.getenv("RESULTS_DIR", "results")
# Create the directory if it doesn't exist, and make the parent directories as well if need be 
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
# output file name that encodes condition and job identity 
outfile <- sprintf("%s/N%d_p%d_%s_%s_block%d_task%s_job%s.csv",
                   out_dir, N, p, model, method, block, task_id, job_id)
# Save our results as CSV
write.csv(res, outfile, row.names = FALSE)
# Print out message for .out log on cluster
cat(sprintf("Wrote %d rows (skipped=%d) to %s\n", nrow(res), skipped, outfile))
