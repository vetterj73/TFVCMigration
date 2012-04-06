# tool to parse regression test files and plot results

from numpy import sqrt, array, clip
import types
import glob
import copy
import os
import pylab
import time

# our runs can be grouped for easier display and tracking
# these groups are identified by strings in their names
# groups of data types:
runTypes = [["Focus", "-4mm"      ,"Through Focus Runs"],
            ["Proj",  "Projective","Projective Transform"],
            ["Cam",   "Camera"    ,"Camera Model"],
            ["Iter",  "Iterative" ,"Iterative (Calibration Drift)"],
            ["Other", ""          ,"Other"]]
# each runType containts a name (e.g. 'Proj'), a search string (e.g. find 'Projective' in file name)
# and a label (e.g. "Projective Transform" a descriptive string for the plots)

# the order of the groups are important, as each type is plotted it is removed from the list
# the empty search string in other collects all unused data files and groups them together 
# so that nothing is missed

#some helper functions:
def popName(List, pattern):
    """
    return a list of items from List which match pattern, pop them from list as it goes
    """
    result = []
    for i in reversed(range(len(List))):
        if List[i].find(pattern) >=0:
            result.append(List.pop(i))
    return result

def parseCsvString(line):
    """
    parse a single line of csv text,
    return a single list
    """
    row = []
    line = line.split(",")
    for item in line:
        item = item.strip()
        try:
            temp = int(item)
        except:
            try:
                temp = float(item)
            except:
                temp = item
        row.append(temp)
    return row

# plot line marker symbols
markers = ['o' , 'D' , 'h' , 'H' , 'p' , '+' , '.' , 's' , '*' , 'd' , '1' , '3' , '4' , '2' , 'v' , '<' , '>' , '^' , ',' , 'x' ]
destDir = "//cyberfs.msp.cyberoptics.com/Projects/CyberStitch/ResultsFolders/RegressionRunCharts/"
srcDir = "C:/CyberStitchRegression/"
os.chdir(srcDir)
dirNames = glob.glob("20*") # assume that all directories of interest start out with 21st century date, assume no stray file have this structure
dirNames.sort()  # just in case they aren't in date order


AllResultNames = []  # track all of the regression test file names
OverallResults = {}
for dirName in dirNames:
    Results = {}
    fns = glob.glob(dirName + "/*.txt")
    for fn in fns:
        fh = open(fn)
        lines = fh.readlines()
        fh.close()
        if len(lines) < 3:
            break  # must be an empty file
        resultName = fn[ fn.index("\\")+1 : -4] # remove dirName and '.txt' from the name
        if AllResultNames.count(resultName) < 1:
            AllResultNames.append(resultName)
        Results[resultName] = {"res":[], "summary":[]}
        temp = lines.pop(0)
        temp = lines.pop(0)
        sectionOne = True
        # the results file consists of two sections
        # section 1 is the per panel results for each fiducial
        # section 2 is the summary section
        for line in lines:
            if line[:4] == " Fid":
                sectionOne = False
            if sectionOne:
                row = parseCsvString(line)
                if type(row[0]) != types.StringType and row[-1] != 1:
                    Results[resultName]["res"].append(row)
                elif ( len(row) > 2 
                      and type(row[1]) == types.StringType 
                      and row[1].find("Panel Processing end time: ") == 0 ): 
                        # parse out the end time, remove leading space
                        t = time.strptime(row[1][26:].strip(),"%m/%d/%Y %I:%M:%S %p")
                        Results[resultName]["endTime"] = time.mktime(t)
                        if not Results[resultName].has_key("startTime"):
                            t = time.strptime(row[0][25:].strip(),"%m/%d/%Y %I:%M:%S %p")
                            Results[resultName]["startTime"] = time.mktime(t)
            else:
                # into the summary section
                if line.find("Xoffset RMS:") > 0:
                    XOffsetStart = line.find("Xoffset RMS:")+ len("Xoffset RMS:")
                    XOffsetEnd = line.find(",", XOffsetStart)
                    Results[resultName]["summary"].append( float( line[XOffsetStart:XOffsetEnd] ) )
                    YOffsetStart = line.find("Yoffset RMS:")+ len("Yoffset RMS:")
                    YOffsetEnd = line.find(",", YOffsetStart)
                    Results[resultName]["summary"].append( float( line[YOffsetStart:YOffsetEnd] ) )
                    Results[resultName]["summary"].append( sqrt(Results[resultName]["summary"][0]**2 + Results[resultName]["summary"][1]**2) )
                if line.find("Average Offset:") == 0:
                    row = line.split()
                    Results[resultName]["summary"].append( float( row[-1] ))
    OverallResults[dirName] = Results

#
nRuns = len(dirNames)

for dirName in dirNames:
    startTime = 1e10
    endTime = 0.
    for runName, Result in OverallResults[dirName].iteritems():
        try:
            if Result['endTime'] > endTime:
                endTime = Result['endTime']
            if Result['startTime'] < startTime:
                startTime = Result['startTime']
        except:
            pass # ignore empty data sets
    OverallResults[dirName]["runTime"] = clip(endTime - startTime, 0, 1e5)
    
#
fig2 = pylab.figure(2, figsize=(16.8,10.5),dpi=75)
pylab.axes([0.1,0.25,0.8,0.65])
xVals = []
yVals = []
for x, dirName in enumerate(dirNames):
    xVals.append(x)
    yVals.append( OverallResults[dirName]["runTime"] )

pylab.plot( xVals, array(yVals)/3600., "-x")

pylab.xticks(range(nRuns), dirNames, rotation=90)
pylab.title("CyberStitchFidTest Overall Regression Execution Time, " )
pylab.ylabel("Time (hours)" )
#limY = pylab.ylim()
pylab.ylim( 0, 12)
pylab.grid()
fig2.savefig(destDir + "Regression_ExecutionTime.png")

pylab.close()


ResultNames = copy.copy(AllResultNames)
pylab.close()
for runName, searchPattern, Label  in runTypes:
    fig2 = pylab.figure(2, figsize=(16.8,10.5),dpi=75)
    pylab.axes([0.1,0.25,0.6,0.65])
    group = popName(ResultNames, searchPattern)
    group.sort()
    for item in group:
        xVals = []
        yVals = []
        for x, dirName in enumerate(dirNames):
            if OverallResults[dirName].has_key(item):
                xVals.append(x)
                yVals.append( OverallResults[dirName][item]['summary'][2] )
        pylab.plot( xVals, yVals, 
            "-"+markers[group.index(item)%len(markers)],
            label=item)
    #
    pylab.legend(loc='right', bbox_to_anchor=(1.35, 0.8),
              ncol=1, fancybox=True, shadow=True,
              prop={"size":10})
    #legend(loc='best' , #bbox_to_anchor=(0.5, 1.0),
    #          ncol=4, fancybox=True, shadow=True,
    #          prop={"size":10})
    pylab.xticks(range(nRuns), dirNames, rotation=90)
    pylab.title("CyberStitchFidTest Regression Results, " + Label )
    pylab.ylabel("RMS Offset (sqrt (XOffsetRMS**2 + YOffsetRMS**2))" )
    limY = pylab.ylim()
    pylab.ylim( 0, limY[1])
    pylab.grid()
    #pylab.show()
    fig2.savefig(destDir + "Regression_" + runName + ".png")
    pylab.close()
