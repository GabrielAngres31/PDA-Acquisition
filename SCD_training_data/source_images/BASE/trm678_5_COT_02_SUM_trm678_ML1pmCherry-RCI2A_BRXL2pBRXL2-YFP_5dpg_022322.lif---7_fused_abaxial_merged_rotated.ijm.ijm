// This macro processes all max projections in a folder to show both channels as a composite.
// It makes the first channel green and the second channel magenta.

inputFile = File.openDialog("Choose your input file");

open(inputFile);
name = substring(getTitle(), 0, lengthOf(getTitle())-4);
run("Stack to Images");
selectImage(name + "-0001");
close();
selectImage(name + "-0002");
run("Grays");
save(substring(inputFile, 0, lengthOf(inputFile)-4) + "-0002" + ".tif")
close();