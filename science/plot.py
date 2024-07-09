from matplotlib import font_manager
import matplotlib.pyplot as plt

def setFonts():
    font_path = 'fonts/LibertinusSerif-Regular.otf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams["mathtext.fontset"] = 'stix'