from tkinter import *

from PIL import Image, ImageTk

window = Tk()
window.title("Stomata Image Classifier")
window.geometry("800x640")
frame = Frame(window)
frame.pack()

# PLACEHOLDER CODE
example_image = Image.open("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD training data\\sample_images\\CAT_3-1468x-1545y-example.tif")

test = ImageTk.PhotoImage(example_image)
label1 = Label(image=test)
label1.image = test

# label1.place(x=300, y=100)

window.resizable(width=False, height=False)

# FRAME CONSTRUCTION

coreframe = Frame(window)
coreframe.pack()

FRAME_imageComparison = Frame(coreframe, bg = "grey")
FRAME_imageComparison.pack(side = TOP, fill = X, expand = False)

FRAME_actionButtons = Frame(coreframe, bg = "yellow")
FRAME_actionButtons.pack(side = BOTTOM, fill = X, expand = False)

SUBFRAME_baseImage = Frame(FRAME_imageComparison, bg = "blue")
SUBFRAME_baseImage.pack(side = LEFT, fill = X, expand = False)

SUBFRAME_annoImage = Frame(FRAME_imageComparison, bg = "orange")
SUBFRAME_annoImage.pack(side = RIGHT, fill = X, expand = False)

SUBFRAME_generatorButtons = Frame(FRAME_actionButtons)
SUBFRAME_generatorButtons.pack(side = LEFT, fill = BOTH, expand = False)

SUBFRAME_decisionsButtons = Frame(FRAME_actionButtons)
SUBFRAME_decisionsButtons.pack(side = RIGHT, fill = BOTH, expand = False)

SUBFRAME_stomataDecisions = Frame(SUBFRAME_decisionsButtons)
SUBFRAME_stomataDecisions.pack(side = TOP, fill = BOTH, expand = False)

SUBFRAME_nothingDecisions = Frame(SUBFRAME_decisionsButtons)
SUBFRAME_nothingDecisions.pack(side = BOTTOM, fill = BOTH, expand = False)

# Button that opens file browser and allows you to select two images to use for selection

BUTTON_selectBase = Button(FRAME_imageComparison, text = "Open Base Image", fg = "black")
BUTTON_selectBase.pack(side=LEFT, fill = X)

BUTTON_selectAnno = Button(FRAME_imageComparison, text = "Open Annotation", fg = "black")
BUTTON_selectAnno.pack(side=LEFT, fill = X)

# Make Image generation button/s

### Unbiased Random
BUTTON_unbiasedRandom = Button(SUBFRAME_generatorButtons, text = "Random Image", fg = "black")
BUTTON_unbiasedRandom.pack(side=TOP, fill = X)

### Filtered Random (avoids totally blank spaces)
BUTTON_filteredRandom = Button(SUBFRAME_generatorButtons, text = "Useful Region", fg = "black")
BUTTON_filteredRandom.pack(side=TOP, fill = X)

### Filtered Target (Looks for sections with annotations)
BUTTON_filteredTarget = Button(SUBFRAME_generatorButtons, text = "Annotated Section", fg = "black")
BUTTON_filteredTarget.pack(side=TOP, fill = X)


# On Image Generation, provide buttons for classification
BUTTON_presentWithPore = Button(SUBFRAME_stomataDecisions, text = "Stomata\n(VISIBLE Pore)", fg = "black")
BUTTON_presentWithPore.pack(side=LEFT, fill = X)

BUTTON_presentWithoutPore = Button(SUBFRAME_stomataDecisions, text = "Stomata\nNOT Visible Pore", fg = "black")
BUTTON_presentWithoutPore.pack(side=LEFT, fill = X)

BUTTON_notPresent = Button(SUBFRAME_nothingDecisions, text = "NOT PRESENT", fg = "black")
BUTTON_notPresent.pack(side=LEFT, fill = X)

BUTTON_unclear = Button(SUBFRAME_nothingDecisions, text = "UNCLEAR", fg = "black")
BUTTON_unclear.pack(side=LEFT, fill = X)

# Confirmation dialogue and saving

# Button to Exit

# GRAB EXAMPLE IMAGE

window.mainloop()




