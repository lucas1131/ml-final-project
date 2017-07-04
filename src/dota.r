source("src/mlp.r")

teams.table = 0
players.table = 0

winner = 0

dota.prepare.datdota.data <- function(dataset.path = "dataset/train.csv"){

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

dota.prepare.opendota.data <- function(dataset.path = "dataset/train.csv"){

	dataset = read.csv(dataset.path, header = TRUE)

	n = nrow(dataset)

	# Discard
	dataset$match_id = NULL

	# Normalize time by 1h
	dataset$duration = dataset$duration/3600

	# Dire = 0 Radiant = 1
	dataset$radiant_win = as.numeric(dataset$radiant_win)-1

	# Get Kills difference between teams
	dataset$score = dataset$radiant_score - dataset$dire_score
	dataset$radiant_score = NULL
	dataset$dire_score = NULL

	# dataset$gold_adv_median = rep(0, n)
	# dataset$gold_adv_mean = rep(0, n)
	# dataset$xp_adv_median = rep(0, n)
	# dataset$xp_adv_mean = rep(0, n)

	# gold = rep(0, n)
	# xp = rep(0, n)

	# for (i in 1:n) {

	# 	# Gold advantages
	# 	gold = unlist(strsplit(as.character(dataset[i, "radiant_gold_adv" ]), "\\[|\\]|,"))
	# 	gold = as.numeric(gold[2:length(gold)])
		
	# 	gold.adv.median = (median(gold) > 0)
	# 	gold.adv.mean = (mean(gold) > 0)

	# 	dataset$gold_adv_median[i] = as.numeric(gold.adv.median)
	# 	dataset$gold_adv_mean[i] = as.numeric(gold.adv.mean)

	# 	# XP advantages
	# 	xp = unlist(strsplit(as.character(dataset[i, "radiant_xp_adv" ]), "\\[|\\]|,"))
	# 	xp = as.numeric(xp[2:length(xp)])
		
	# 	xp.adv.median = (median(xp) > 0)
	# 	xp.adv.mean = (mean(xp) > 0)

	# 	dataset$xp_adv_median[i] = as.numeric(xp.adv.median)
	# 	dataset$xp_adv_mean[i] = as.numeric(xp.adv.mean)
	# }

	dataset[is.na.data.frame(dataset)] = 0

	dataset$radiant_gold_adv = NULL
	dataset$radiant_xp_adv = NULL

	# Extract Winner table for validating
	winner <<- dataset$radiant_win
	dataset$radiant_win = NULL

	return (dataset)
}

dota.prepare.data <- function(dataset.path = "dataset/train.csv", opendota=TRUE){

	if(opendota == TRUE)
		return (dota.prepare.opendota.data(dataset.path))
	else 
		return (dota.prepare.datdota.data(dataset.path))
}

# dota.test <- function(mlp, test.path = "dataset/test.csv", validation.path){
dota.test <- function(mlp, testset, winners){

	test.size = nrow(testset)

	ret = list()
	ret$results = rep(0, test.size)

	for(i in 1:test.size){
		ret$results[i] = mlp.forward(mlp, testset[i,])$f.output # We just want the output
	}

	ret$binary.results = dota.discretize.results(ret$results)
	ret$accuracy = mlp.accuracy(ret$binary.results, winners)
	
	cat("Accuracy: ", ret$accuracy, "\n")

	return (ret)
}

dota <- function(dataset.path = "dataset/train.csv", size=20, 
	step=0.4, threshold=0.15){

	dataset = as.matrix(dota.prepare.data(dataset.path))

	# size = mlp.upper.hidden.size(input.size=ncol(dataset)-1, output.size=1, 
	# 	n.samples=nrow(dataset), alpha=alpha)
	mlp = mlp.create(input.size=ncol(dataset), output.size=1, hidden.size=size)

	# Randomize dataset for random validation
	mlp$rand.indexes = sample(nrow(dataset))

	# Use last 10% for validation
	# mlp = mlp.train(mlp, dataset[mlp$rand.indexes[1:400],], 
	# 	as.matrix(winner[ mlp$rand.indexes[1:400] ]), 
	# 	step=step, threshold=threshold)

	mlp = mlp.train(mlp, dataset[mlp$rand.indexes[1:13000],], 
		as.matrix(winner[ mlp$rand.indexes[1:13000] ]), 
		step=step, threshold=threshold)

	return (mlp)
}

dota.discretize.results <- function(results){ 
	ret = rep(0, length(results))
	ret[results >= 0.5] = TRUE
	return (ret)
}
