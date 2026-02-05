import toga
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, Pack
from PIL import Image
import typing as tp

class AnnotationHelper(toga.App):

    base_filepath :str = "SCD_training_data/base_placeholder.png"
    annot_filepath:str = "SCD_training_data/annot_placeholder.png"

    base_file :Image = Image.open(base_filepath)
    annot_file:Image = Image.open(annot_filepath)

    window_size = 64

    corner_x = 0
    corner_y = 0

    current_stoma_number:int = 0

    value_slider_opacity = 155

    def action_loadfile_image(self, widget):
        print("a1 placeholder")
    
    def action_loadfile_csv  (self, widget):
        print("a2 placeholder")
    
    def action_loadfile_info (self, widget):
        print("a3 placeholder")
    
    button_playtest_field = toga.Box()
    
    async def get_file_handler_base(self, widget):
        try:
            fname = await self.dialog(toga.OpenFileDialog("Test File Opener"))
            if fname is not None: 
                self.base_filepath = fname
                self.base_file = Image.open(fname)
        except ValueError: pass

    async def get_file_handler_annot(self, widget):
        try:
            fname = await self.dialog(toga.OpenFileDialog("Test File Opener"))
            if fname is not None: 
                self.annot_filepath = fname
                self.annot_file = Image.open(fname)
        except ValueError: pass

    def startup(self):

        # Commands
        file_commands = toga.Group("Commands")
        
        # File Getters
        get__annot_file = toga.Command(self.get_file_handler_annot, 
                                text = "Annot Handler",
                                tooltip = "this is a test button",
                                group = file_commands)
        get__base_file = toga.Command(self.get_file_handler_base, 
                                text = "Base Handler",
                                tooltip = "this is a test button",
                                group = file_commands)
        
        # Location of the Display Images

        panel_miniimages = toga.Box(style=Pack(direction=ROW, padding_top=20, padding=10))

        # Buttons
        def stoma_number_increment(self):
            self.text_input.value += 1

        self.text_input_id = toga.TextInput(on_confirm = None, validators=None)
        self.text_input_id.value = 1

        def stoma_number_increment(widget):
            self.text_input_id.value = int(self.text_input_id.value)+1
        
        def stoma_number_decrement(widget):
            new_val = int(self.text_input_id.value)-1
            if new_val+1:
                self.text_input_id.value = int(self.text_input_id.value)-1
            else: pass

        self.button_increment = toga.Button("<<", on_press=stoma_number_decrement)

        self.button_decrement = toga.Button(">>", on_press=stoma_number_increment)

        value_opacity = toga.TextInput(readonly=True, value=155)

        def update_opacity(slider):
            self.value_slider_opacity = slider.value
            value_opacity.value = slider.value

        slider_opacity = toga.Slider(min=0, max=255, value=155, tick_count=256, on_release=update_opacity, on_change=update_opacity)

        self.test_canvas = toga.Canvas(style=Pack(flex=1))
        self.test_canvas.Fill(color=toga.colors.BLACK)


        # container = toga.OptionContainer(
        #     content=[
        #         ("Mini Images", panel_miniimages)

        #     ],
        # )

        panel_miniimages.add(self.button_increment)
        panel_miniimages.add(self.text_input_id)
        panel_miniimages.add(self.button_decrement)
        panel_miniimages.add(slider_opacity)
        panel_miniimages.add(value_opacity)
        panel_miniimages.add(self.test_canvas)

        # self.main_window.add(self.testbutton)
        self.main_window = toga.MainWindow()
        # self.main_window.content = container
        self.main_window.toolbar.add(get__base_file, get__annot_file)
        self.main_window.content = panel_miniimages
        self.main_window.show()

def main():
    return AnnotationHelper("Annotation Helper", "Object")

if __name__ == "__main__":
    main().main_loop()
