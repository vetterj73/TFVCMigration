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
            ["Populated", "Populated", "Populated compared to Non-Populated"],
            ["Edge",  "Edge"      ,"Apply Board Edge Detection"],
            ["DEK-VG","DEK-VG"    ,"DEK-VG with and without Calibration Error"],
            ["Proj",  "Projective","Projective Transform"],
            ["Cam",   "Camera"    ,"Camera Model"],
            ["Iter",  "Iterative" ,"Iterative (Calibration Drift)"],
            ["Other", ""          ,"Other"]]

# Oldest date to plot
ignoreBefore = time.strptime("2012-04-23","%Y-%m-%d")
# Data sampling interval for charting
samplingInterval = 3
# Number of latest datasets need to keep for charting 
lastNumber = 10 
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
# where to place plots ** this must match the location in the RunChart.htm web page
destDir = "//cyberfs.msp.cyberoptics.com/Projects/CyberStitch/RegressionRunCharts/"
# destDir = "C://temp//RunCharts//"
# location of regression test result text files ** use the projects directory **
#srcDir = "C:/CyberStitchRegression/"
srcDir = "//cyberfs.msp.cyberoptics.com/Projects/CyberStitch/CyberStitchRegression/"
os.chdir(srcDir)
dirNames = glob.glob("20*") # assume that all directories of interest start out with 21st century date, assume no stray file have this structure
dirNames.sort()  # just in case they aren't in date order
# remove old data
for i, dirName in enumerate(dirNames):
    t=time.strptime(dirName,"%Y-%m-%d-%H-%M-%S")
    if t > ignoreBefore:
        break

dirNames = dirNames[i:]
# read all of the regression files
AllResultNames = []  # track all of the regression test file names
OverallResults = {}  # a dictionary to hold the results.
for dirName in dirNames: # walk through each regression test (by date)
    Results = {}
    Results["panelsPerRun"] = 0
    fns = glob.glob(dirName + "/*.txt")
    for fn in fns:
        fh = open(fn)
        lines = fh.readlines()
        fh.close()
        if len(lines) < 3:
            break  # must be an empty file
        resultName = fn[ fn.index("\\")+1 : -4] # remove dirName and '.txt' from the name
        if AllResultNames.count(resultName) < 1:  # a new run?  e.g. add the name '3FidsProjective' to the list
            AllResultNames.append(resultName)
        Results[resultName] = {"res":[], "summary":[], "times":[]}
        temp = lines.pop(0) # remove first two lines
        temp = lines.pop(0)
        sectionOne = True
        # the results file consists of two sections
        # section 1 is the per panel results for each fiducial
        # section 2 is the summary section
        xSquaredSum = 0
        ySquaredSum = 0
        numFidsFound = 0
        worstFid = 0
        for line in lines:
            if line[:4] == " Fid":
                sectionOne = False
            if sectionOne:
                row = parseCsvString(line)
                if type(row[0]) != types.StringType and row[-1] != 1:
                    Results[resultName]["res"].append(row) # save individual fid results, 
                    xSquaredSum += row[4]**2
                    ySquaredSum += row[5]**2
                    offset = sqrt( row[4]**2 + row[5]**2 )
                    if offset > worstFid:
                        worstFid = offset
                    numFidsFound += 1
                    # we don't yet plot this detailed data
                elif ( len(row) > 2 
                      and type(row[1]) == types.StringType 
                      and row[1].find("Panel Processing end time: ") == 0 ): 
                        Results["panelsPerRun"] += 1
                        # parse out the end time, remove leading space
                        tEnd = time.strptime(row[1][26:].strip(),"%m/%d/%Y %I:%M:%S %p")
                        tStart = time.strptime(row[0][25:].strip(),"%m/%d/%Y %I:%M:%S %p")
                        Results[resultName]["endTime"] = time.mktime(tEnd)  # this will finish with the last end time
                        Results[resultName]["times"].append( [tStart, tEnd] )
                        if not Results[resultName].has_key("startTime"):
                            Results[resultName]["startTime"] = time.mktime(tStart) # only save the first start time
            else:
                # into the summary section, this is where most of the plot data is found
                if line.find("Xoffset RMS:") > 0:
                    XOffsetStart = line.find("Xoffset RMS:")+ len("Xoffset RMS:")
                    XOffsetEnd = line.find(",", XOffsetStart)
                    #Results[resultName]["summary"].append( float( line[XOffsetStart:XOffsetEnd] ) )
                    if numFidsFound >= 1:
                        Results[resultName]["summary"].append( sqrt(xSquaredSum / numFidsFound) )
                    else:
                        Results[resultName]["summary"].append(0.)
                    YOffsetStart = line.find("Yoffset RMS:")+ len("Yoffset RMS:")
                    YOffsetEnd = line.find(",", YOffsetStart)
                    #Results[resultName]["summary"].append( float( line[YOffsetStart:YOffsetEnd] ) )
                    if numFidsFound >= 1:
                        Results[resultName]["summary"].append( sqrt(ySquaredSum / numFidsFound) )
                    else:
                        Results[resultName]["summary"].append(0.)
                    Results[resultName]["summary"].append( sqrt(Results[resultName]["summary"][0]**2 + Results[resultName]["summary"][1]**2) )
                    Results[resultName]["summary"].append( worstFid )
                if line.find("Average Offset:") == 0:
                    row = line.split()
                    Results[resultName]["summary"].append( float( row[-1] ))
    OverallResults[dirName] = Results


#
nRuns = len(dirNames)

# find the total execution time -- use the time from the earliest start time to the last end time
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

#Do data sampling (Current setting: every 4th point plus all of the last 10))
dirSampleNames = []
if (len(dirNames) > lastNumber):
    for i in range (0,len(dirNames)-lastNumber,samplingInterval):
        dirSampleNames.append(dirNames[i])
    for i in range (len(dirNames)-lastNumber,len(dirNames)):
        dirSampleNames.append(dirNames[i])
nSampleRuns = len(dirSampleNames)   
#
#plotBoxDims = [0.1,0.25,0.6,0.65]
plotBoxDims = [0.08,0.25,0.6,0.65]
# position the plot within the window
# 0.1 from left is enough room for y label
# 0.25 from botom is needed for very large x tick labels
# 0.6 is active width of graph, lots of room at right for legend
# 0.65 is active height (saves some room at top for title)

# plot out the total run time
xVals = [] # since every regression tset should have a time I could have written xVals = range(nRuns)
timeVals = []
panelCounts = []
for x, dirName in enumerate(dirSampleNames):
    xVals.append(x)
    timeVals.append( OverallResults[dirName]["runTime"] )
    panelCounts.append( OverallResults[dirName]["panelsPerRun"] )
fig2 = pylab.figure(2, figsize=(16.8,10.5),dpi=75)
# results in a 1680 x 1050 plot with reasonalble text sizes
ax1 = pylab.axes(plotBoxDims) 
pylab.plot( xVals, array(timeVals)/3600., "b-x", label="Execution Time (Hours)")
pylab.xticks(range(nSampleRuns), dirSampleNames, rotation=90) # do this twice????
pylab.title("CyberStitchFidTest Overall Regression Execution Time and Panel Count" )
pylab.ylabel("Time (hours)" , color='b' )
pylab.ylim( 0, 12)
ax2 = ax1.twinx() # second y axis
pylab.plot( xVals, panelCounts, "g-x", label="Panel Count")
limY = pylab.ylim()
pylab.ylim( 0, limY[1]*1.1)
pylab.xticks(range(nSampleRuns), dirSampleNames, rotation=90)
pylab.ylabel("Total Panels" , color='g')
#pylab.legend()
#legends for both (if desired) need be done one at a time and manually positioned
pylab.grid()
for tl in ax1.get_yticklabels():
    tl.set_color('b')
for tl in ax2.get_yticklabels():
    tl.set_color('g')
fig2.savefig(destDir + "Regression_ExecutionTime.png")
pylab.close()

# Plot RMS Errors
ResultNames = copy.copy(AllResultNames)
pylab.close()
for runName, searchPattern, Label  in runTypes:
    fig2 = pylab.figure(2, figsize=(16.8,10.5),dpi=75)
    pylab.axes(plotBoxDims)
    group = popName(ResultNames, searchPattern) # only those file names matching the pattern
    group.sort()
    for item in group:
        xVals = []
        yVals = []
        for x, dirName in enumerate(dirSampleNames):
            if OverallResults[dirName].has_key(item):
                xVals.append(x)
                yVals.append( OverallResults[dirName][item]['summary'][2] )
        pylab.plot( xVals, yVals, 
            "-"+markers[group.index(item)%len(markers)],
            label=item)
    #
    pylab.legend(loc='right', bbox_to_anchor=(1.38, 0.8),
              ncol=1, fancybox=True, shadow=True,
              prop={"size":10})
    #legend(loc='best' , #bbox_to_anchor=(0.5, 1.0),
    #          ncol=4, fancybox=True, shadow=True,
    #          prop={"size":10})
    pylab.xticks(range(nSampleRuns), dirSampleNames, rotation=90)
    pylab.title("CyberStitchFidTest Regression Results, " + Label )
    pylab.ylabel("RMS Offset (sqrt (XOffsetRMS**2 + YOffsetRMS**2))" )
    limY = pylab.ylim()
    pylab.ylim( 0, limY[1])
    pylab.grid()
    #pylab.show()
    fig2.savefig(destDir + "Regression_" + runName + ".png")
    pylab.close()

# Plot Worst Case Errors
ResultNames = copy.copy(AllResultNames)
pylab.close()
for runName, searchPattern, Label  in runTypes:
    fig2 = pylab.figure(2, figsize=(16.8,10.5),dpi=75)
    pylab.axes(plotBoxDims)
    group = popName(ResultNames, searchPattern) # only those file names matching the pattern
    group.sort()
    for item in group:
        xVals = []
        yVals = []
        for x, dirName in enumerate(dirSampleNames):
            if OverallResults[dirName].has_key(item):
                xVals.append(x)
                yVals.append( OverallResults[dirName][item]['summary'][3] )
        pylab.plot( xVals, yVals, 
            "-"+markers[group.index(item)%len(markers)],
            label=item)
    #
    pylab.legend(loc='right', bbox_to_anchor=(1.38, 0.8),
              ncol=1, fancybox=True, shadow=True,
              prop={"size":10})
    #legend(loc='best' , #bbox_to_anchor=(0.5, 1.0),
    #          ncol=4, fancybox=True, shadow=True,
    #          prop={"size":10})
    pylab.xticks(range(nSampleRuns), dirSampleNames, rotation=90)
    pylab.title("CyberStitchFidTest Regression Results - WORST CASE FIDS, " + Label )
    pylab.ylabel("Worst Fiducial Offset (um)" )
    limY = pylab.ylim()
    pylab.ylim( 0, limY[1])
    pylab.grid()
    #pylab.show()
    fig2.savefig(destDir + "Regression_WorstCase" + runName + ".png")
    pylab.close()
