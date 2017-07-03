source("src/mlp.r")

dota.prepare.data <- function(dataset.path = "dataset/train.csv"){

	dataset = read.csv(dataset.path, header = TRUE)

	dataset$Start.Date.Time = NULL
	dataset$Match.Length = NULL

	return (dataset)
}

dota.test <- function(mlp, test.path = "dataset/test.csv", validation.path){

	testset = as.matrix(dota.prepare.data(test.path))
	test.validate = as.matrix(read.csv(file=validation.path, header=TRUE))

	test.size = nrow(testset)
	testset.use = testset[, 2:ncol(testset)]

	ret = list()
	ret$results = rep(0, test.size)

	for(i in 1:test.size){
		tmp = mlp.forward(mlp, testset.use[i,])
		ret$results[i] = tmp$f.output # We just want the output
	}

	ret$binary.results = dota.discretize.results(results)
	ret$accuracy = mlp.accuracy(ret$binary.results, test.validate[,2])
	cat("Accuracy: ", ret$accuracy, "\n")

	return (ret)
}

dota <- function(dataset.path = "dataset/train.csv", alpha=10, 
	step=0.004, threshold=1e-1){

	dataset = as.matrix(dota.prepare.data(dataset.path))

	size = mlp.upper.hidden.size(input.size=ncol(dataset)-1, output.size=1, 
		n.samples=nrow(dataset), alpha=alpha)
	mlp = mlp.create(input.size=ncol(dataset)-2, output.size=1, hidden.size=size)
	mlp = mlp.train(mlp, dataset[,2:(ncol(dataset)-1)], dataset[,ncol(dataset)], 
		step=step, threshold=threshold)

	return (mlp)
}
