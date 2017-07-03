source("src/mlp.r")

teams.table = 0
players.table = 0

winner = 0

dota.prepare.data <- function(dataset.path = "dataset/train.csv"){

	dataset = read.csv(dataset.path, header = TRUE)

	n = nrow(dataset)

	# Discard
	dataset$Start.Date.Time = NULL
	dataset$Match.Length = NULL
	dataset$Match.ID = NULL
	dataset$Series = NULL
	dataset$League = NULL
	dataset$Dummy = NULL

	# Normalize time by 1h
	dataset$Duration.secs = dataset$Duration.secs/3600

	# Dire = 0 Radiant = 1
	dataset$Winner = as.numeric(dataset$Winner)-1

	# Get Kills difference between teams
	tmp = strsplit(as.character(dataset$Kills.Score), "-")
	dataset$Kills = rep(0, n)

	for (i in 1:n) {
		tmp2 = as.numeric(unlist(tmp[i]))

		dataset$Kills[i] = tmp2[1] - tmp2[2]
	}

	dataset$Kills.Score = NULL

	# Get all player names
	player.names = unique(c(
					unique(unlist(strsplit(levels(dataset$Radiant.Players), ", "))),
	 				unique(unlist(strsplit(levels(dataset$Dire.Players), ", ")))
	 				)
				)

	# Get all team names
	team.names = unique(c(
					unique(levels(dataset$Radiant.Team)),
	 				unique(levels(dataset$Dire.Team))
	 				)
				)

	# Generate global lookup table
	players.table <<- factor(player.names, levels=player.names, labels=player.names)
	teams.table <<- factor(team.names, levels=team.names, labels=team.names)

	# Map Radiant team name string to number in global teams.table
	tmp = rep(0, n)
	for (i in 1:n) {
		tmp[i] = which(team.names == dataset$Radiant.Team[i])
	}
	dataset$RTeam = tmp

	# Map Dire team name string to number in global teams.table
	for (i in 1:n) {
		tmp[i] = which(team.names == dataset$Dire.Team[i])
	}
	dataset$DTeam = tmp

	dataset$Radiant.Team = NULL
	dataset$Dire.Team = NULL

	# For testing, drop player names (small difference with smal dataset)
	dataset$Radiant.Players = NULL
	dataset$Dire.Players = NULL

	# Extract Winner table for validating
	winner <<- dataset$Winner
	dataset$Winner = NULL

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

dota <- function(dataset.path = "dataset/train.csv", alpha=5, 
	step=0.4, threshold=1e-1){

	dataset = as.matrix(dota.prepare.data(dataset.path))

	size = mlp.upper.hidden.size(input.size=ncol(dataset)-1, output.size=1, 
		n.samples=nrow(dataset), alpha=alpha)
	mlp = mlp.create(input.size=ncol(dataset)-2, output.size=1, hidden.size=size)

	# Randomize dataset for random validation
	dataset = as.matrix(dataset[sample(nrow(dataset)),])

	# Use last 10% for validation
	mlp = mlp.train(mlp, dataset[1:900,2:(ncol(dataset)-1)], as.matrix(winner[1:900]), 
		step=step, threshold=threshold)

	return (mlp)
}
