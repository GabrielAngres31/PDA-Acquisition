library("Rcmdr")

Dataset <- 
  read.table("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/AZD_test/concatenated.csv",
             header=TRUE, stringsAsFactors=TRUE, sep=",", na.strings="NA", dec=".", 
             strip.white=TRUE)