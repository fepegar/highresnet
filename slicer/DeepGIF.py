import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import vtk, qt, ctk, slicer
import sitkUtils as su
from slicer.ScriptedLoadableModule import *



class DeepGIF(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Deep GIF"
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Fernando Perez-Garcia (University College London)"]
    self.parent.helpText = """
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
""" # replace with organization, grant and thanks.


class DeepGIFWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.makeGUI()

  def makeGUI(self):
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

<<<<<<< HEAD
=======

>>>>>>> 64efae71168a43a093a6a1f7645f85f2e09ec4f9
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.clicked.connect(self.onApplyButton)

    # Add vertical spacer
    self.layout.addStretch(1)

  def onSelect(self):
    self.applyButton.enabled = True

  def onApplyButton(self):
    logic = DeepGIFLogic()



class DeepGIFLogic(ScriptedLoadableModuleLogic):

  def segment(self, inputVolumeNode):
    self.ensureLibraries()

  def ensureLibraries(self):
    try:
      import highresnet
    except ImportError:
      slicer.util.pip_install('torch>=1.2')  # for PyTorch Hub
      slicer.util.pip_install('highresnet')


class DeepGIFTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_DeepGIF1()

  def test_DeepGIF1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    volumeNode = SampleData.SampleDataLogic().downloadMRHead()
    self.delayDisplay('Finished with download and loading')

    logic = DeepGIFLogic()
<<<<<<< HEAD
    segmentationNode = logic.segment(volumeNode)
=======
    labelMapNode = logic.segment(volumeNode)
>>>>>>> 64efae71168a43a093a6a1f7645f85f2e09ec4f9
    self.delayDisplay('Test passed!')
