#################################################################################################
# SCC0233 - Aplicações de Aprendizado de Máquina e Mineração de Dados							#
# Professor: Rodrigo Fernandes de Mello															#
# Aluno: Rafael Miranda Lopes																	#
# nUSP: 6520554																					#
#-----------------------------------------------------------------------------------------------#
# Kaggle - Digit Recognizer																		#
# 	Para executar, rode a função mnist.run, com os argumentos conforme especificados na própria	#
# função.																						#
#################################################################################################

#---------------------------------------------- MLP --------------------------------------------#
f <- function(net) {
	return (1/(1+exp(-net)))
}

df_dnet <- function(net) {
	return (f(net) * (1-f(net)))
}

mlp <- function(input.length, hidden.length, output.length) {
	cat("I: ", input.length, "\nH: ", hidden.length, "\nO: ", output.length, "\n")
	mlp = list()
	mlp$input.length = input.length
	mlp$hidden.length = hidden.length
	mlp$output.length = output.length
	#pesos entre input e camada escondida (+1 para bias) - (input+1) x (hidden)
	mlp$w_i_h = matrix(runif(min = -0.5, max = 0.5, n = (input.length+1)*hidden.length),
		nrow = input.length+1, ncol = hidden.length)
	#pesos entre camada escondida e a de saída (+1 para bias) - (hidden+1) x (output)
	mlp$w_h_o = matrix(runif(min = -0.5, max = 0.5, n = (hidden.length+1)*output.length),
		nrow = hidden.length+1, ncol = output.length)
		
	return(mlp)
}

mlp.forward <- function(mlp, x) {
	#calcula net da camada escondida
	x = c(as.vector(x),1)
	netH = as.vector(x %*% mlp$w_i_h)
	#calcula net da camada de saída
	h = c(f(netH),1)
	netO = as.vector(h %*% mlp$w_h_o)
	#Return
	mlp.forward = list()
	mlp.forward$netH = netH
	mlp.forward$netO = netO
	return (mlp.forward)
}

mlp.train <- function(X, Y, H.length, eta = 0.1, threshold = 1e-2) {
	Y = as.matrix(Y)
	X = as.matrix(X)
	mlp = mlp(ncol(X), H.length, ncol(as.matrix(Y)))
	sqerror = threshold*2
	sqerror.last = sqerror*1024
	epoch = 1
	while (sqerror > threshold) {
		sqerror = 0
		#para cada conjunto de entradas-saídas de treinamento - backpropagation
		for (i in 1:nrow(X)){
			fwd = mlp.forward(mlp, as.vector(X[i,]))
			# delta = saída esperada - output
			deltas = Y[i,] - f(fwd$netO)
			#Camada de saída
			deltas_o = deltas*df_dnet(fwd$netO)
			w_h_o = mlp$w_h_o + eta*( (as.matrix( c(f(fwd$netH),1) )) %*% t(as.matrix(deltas_o)) )
			#Camada escondida
			deltas_h = df_dnet(fwd$netH) * ( t(as.matrix(deltas_o)) %*%
				t(as.matrix(mlp$w_h_o[1:mlp$hidden.length,])) )
			w_i_h = mlp$w_i_h + eta * ( ( as.matrix( c( X[i,] ,1) )) %*% (as.matrix(deltas_h)) )
			#finaliza atualização dos pesos
			mlp$w_h_o = w_h_o
			mlp$w_i_h = w_i_h
			#Soma dos quadrados dos erros
			sqerror = sqerror + sum(deltas^2)	
		}
		sqerror = sqerror/nrow(X)
		cat("Epoch: ", epoch, "| Avg sqerror: ", sqerror, "\n")
		flush.console()
		#Atualiza eta dinamicamente
		if(sqerror >= sqerror.last) {
			eta = eta * 0.8
			cat("New eta: ", eta, "\n")
		}
		sqerror.last = sqerror
		epoch = epoch+1
	}
	return (mlp)
}

#---------------------------------------------- PCA --------------------------------------------#
pca <- function(dataset){
	#Subtrai as médias das colunas (centraliza os valores nas direções)
	dataset = scale(dataset, center = TRUE, scale = FALSE)
	dataset.var = var(dataset)
	pca = list()
	pca$eigenV = eigen(dataset.var, symmetric = TRUE) #$values e $vectors
	pca$vari = abs(pca$eigenV$values)
	return (pca)
}

pca.transform <- function(pca, dataset){
	dataset = scale(dataset, center = TRUE, scale = FALSE)
	#Scale alternativo para o mnist
	dataset = dataset/255
	y = as.matrix(dataset) %*% as.matrix(pca$eigenV$vectors)
	return (y)
}

#---------------------------------------------- Dota --------------------------------------------#
bindfiles <- function(){
	directory = "Dota/"
	files = list.files(path = directory, pattern = "*.csv", full.names = TRUE, recursive = FALSE)
	dataset = read.csv(files[[1]], header = TRUE)
	for(i in 2:length(files)){
		ds = read.csv(files[[i]], header = TRUE)
		dataset = rbind (dataset, ds)
	}
	return (dataset)
}

retrieve.roles <- function(data.row){
	dota.roles = c("Carry", "Disabler", "Initiator", "Jungler", "Support", "Durable", "Nuker", "Pusher", "Escape")
	#roles = unlist( strsplit(data.row[3:12], ",") )
	roles = (lapply(data.row[3:12], function(x){return(unlist( strsplit(x, ",") ))}))
	#print(roles)
	mlp.row = rep(0, 20)
	mlp.row[1] = as.numeric(data.row[[2]])
	#print(data.row[[2]])
	# 9 roles, 2 teams
	for(i in 1:9){ 
		mlp.row[i+1] = sum(unlist(roles[1:5]) == dota.roles[i])
		mlp.row[i+10] = sum(unlist(roles[6:10]) == dota.roles[i])
	}
	mlp.row[20] = as.numeric(data.row[[13]]) #classe
	#print(as.vector(mlp.row))
	return (as.vector(mlp.row))
}

#Recebe tabela match_id, patch, roles, player_slot, win
format.table <- function(dataset){
	newdf = as.data.frame(matrix(ncol = 13, nrow = nrow(dataset)/10)) # 10 players por match
	names(newdf) = c("MatchID", "Patch", 
		"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "Result")
	#newdf = as.matrix(newdf)
	for(i in 1:(nrow(dataset)/10)){
		newdf[i, 1:2] = dataset[i*10, 1:2] #match_id, patch
		result = dataset[i*10, 5] #result (win)
		if(result == "true")
			newdf[i, 13] = 1
		else
			newdf[i, 13] = 0
		#passa roles para colunas, retirando caracteres desnecessários
		newdf[i, 3:12] = gsub("\\[|\\]|\"", "", dataset[(i*10-9):(i*10), 3])
	}
	#return (newdf)
	mlp.ready.table = t(apply(newdf, 1, retrieve.roles))
	return (mlp.ready.table)
}

dota.test <- function(mlp, testset){
	classes = rep(0, nrow(testset))
	#testset.formated = format.table(testset)
	#classes = apply ( testset, 1, function(x){return ( f(mlp.forward(mlp, x)$netO) ) } )
	for(i in 1:length(classes)){
		classes[i] = f(mlp.forward(mlp, as.vector(testset[i,]/10) )$netO)
	}
	return (classes)
}

dota.run <- function(train, hidden.length, eta = 0.1, threshold = 1.0e-1){
	dota = format.table(train)
	mlp = mlp.train(dota[,2:19]/10, dota[,20], hidden.length, eta, threshold)
	return (mlp)
}































#--------------------------------------------- mnist -------------------------------------------#
mnist.format_output <- function(Y) {
	out = matrix(0L, nrow = length(Y), ncol = 10)
	# 0 ... 9
	for (i in 1:length(Y)){
		out[i,Y[i]+1] = 1
	}
	return (out)
}

mnist.test <- function(mlp, dataset) {
	classes = matrix(0L, nrow = nrow((dataset)), ncol = 1)
	for (i in 1:nrow(dataset)){
		classes[i] = which.max(f(mlp.forward(mlp, dataset[i,])$netO))-1 # -1: 0 ... 9
	}
	return (classes)
}

mnist.compare <- function (expected, output) {
	expected = as.vector(expected)
	output = as.vector(output)
	print (expected)
	print (output)
	ret = list()
	ret$count = sum(expected == output)
	ret$ratio = ret$count/length(expected)
	return (ret)
}

mnist.setup <- function (train.set, test.set){
	# Treino
	train.pixels = train.set[,2:ncol(train.set)]
		#Identifica e remove colunas de variância zero
	varZero = which(apply(train.pixels, 2, var) == 0) #identifica
	train.pixels = train.pixels[-c(varZero)] #remove
	#custom for: pca = prcomp(train.pixels, center = TRUE, scale. = TRUE)
	pca = pca(train.pixels)
	#custom for: train.transf = predict(pca, newdata = train.pixels)
	train.transf = pca.transform(pca, train.pixels)
	train.expected = mnist.format_output(train.set[,1])
	# Teste
	#custom for: test.transf = predict(pca, newdata = test.set[-c(varZero)])
	test.transf = pca.transform(pca, test.set[-c(varZero)])
	ret = list()
	ret$pca = pca
	ret$train.transf = train.transf
	ret$train.expected = train.expected
	ret$test.transf = test.transf
	return(ret)
}

mnist.write.output <- function (classes, filename) {
	df = data.frame(seq( along.with = classes[,1]), classes[,1])
	names(df) = c("ImageId", "Label")
	write.csv(df, filename, row.names = FALSE)
}

mnist.run <- function(train, test, hidden.length, pca.repr = 0.90, eta = 0.1, threshold = 1.0e-2){
	#train é o dataframe de treino (mnist, com label e pixels)
	#test é o dataframe de teste (pixels)
	#hidden.length é o número de neurônios na camada escondida
	#pca.repr é a proporção da variância total dos dados que deve ser englobada pelos PCs
	#eta é a taxa de aprendizado
	#threshold é o erro quadrático médio a que se deseja chegar
	mnistSetup = mnist.setup(train, test)
	#Calcula número de PCs
	#custom for: pca.var = mnistSetup$pca$sdev^2
	pca.var = mnistSetup$pca$vari
	pca.sum = 0
	pca.sum_total = sum(pca.var)
	pca.pcs = 0
	for(i in 1:length(pca.var)){
		pca.sum = pca.sum + pca.var[i]
		if(pca.sum/pca.sum_total > pca.repr) {
			pca.pcs = i
			break
		}
	}
	cat("PCs: ", pca.pcs, "\n")
	flush.console()
	#Treina mlp
	mlp = mlp.train(mnistSetup$train.transf[,1:pca.pcs], 
		mnistSetup$train.expected, hidden.length, eta, threshold)
	#Executa teste sobre o conjunto de testes
	test = mnist.test(mlp, mnistSetup$test.transf[,1:pca.pcs])
	ret = list()
	ret$pca.pcs = pca.pcs #componentes principais
	ret$mlp = mlp #mlp treinada
	ret$test = test	#classes do conjunto de testes
	return(ret)
}
