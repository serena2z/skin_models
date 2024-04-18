## QuickSegment

QuickSegment is a simple tool for basic image segmentation.
It consists of a single HTML file which can be dragged and dropped into a directory containing the images a user wishes to label.
As such, it is completely cross-platform, and its only dependency is a modern browser with Javascript enabled.

### Usage Notes

In some browsers, the Export button will automatically save the generated CSV file to the user's `Downloads` folder, without prompting or presenting "Save As" dialog.
This is unfortunately a browser setting that is out of the control of any web page.

### Export Format

Segmentation data is exported in a CSV format.
Along with the file name, each entry contains three 2D vectors labeled `trns`, `v1`, and `v2`, which describe a (possibly rotated) rectangle within the image.
The `trns` vector gives a translational offset from the origin (top left of the image).
`v1` and `v2` give two orthogonal sides of the rectangle, relative to `trns`.

Currently, the tool only supports grid-aligned rectangles, but this export format is designed to easily extend to rotated rectangles in the future.

### Future features
- Support for segmenting image data in additional file formats (e.g. NIfTI and DICOM)
- Support for rotated selections
