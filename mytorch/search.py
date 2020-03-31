import numpy as np


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    PathScore = dict()
    BlankPathScore = dict()
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializaePaths(SymbolSets, y_probs[:,0])
    print(NewPathsWithTerminalSymbol)
    print(NewPathScore)
    for t in range(1, y_probs.shape[1]):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)
        print(PathsWithTerminalSymbol)
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t], BlankPathScore, PathScore)
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:,t],BlankPathScore, PathScore)
    
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
   # To-DO
    BestPath = max(FinalPathScore.items(), key = lambda k:k[1])[0]
    # print(FinalPathScore)

    return BestPath, FinalPathScore

def InitializaePaths(SymbolSets, y):
    InitialBlankPathScore = dict()
    InitialPathScore = dict()
    path = ''
    InitialBlankPathScore[path] = y[0]
    InitialPathsWithFinalBlank = {path}
    #Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathsWithFinalSymbol = set()
    for c in range(len(SymbolSets)):
        path = SymbolSets[c]
        InitialPathScore[path] = y[c + 1]
        InitialPathsWithFinalSymbol.add(path)
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = dict()
    for path in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]
    
    for path in PathsWithTerminalSymbol:
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]

    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = dict()
    for path in PathsWithTerminalBlank:
        for c in range(len(SymbolSets)):
            newpath = path + SymbolSets[c]
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[c + 1]
    
    for path in PathsWithTerminalSymbol:
        for c in range(len(SymbolSets)):
            if (SymbolSets[c] == path[-1]):
                newpath = path
            else:
                newpath = path + SymbolSets[c]
            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath] += PathScore[path] * y[c + 1]
            else:
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[c + 1]

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = dict()
    PrunedPathScore = dict()
    scorelist = []
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p][0])
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p][0])
    scorelist = sorted(scorelist, reverse = True)
    print(scorelist)
    if BeamWidth < len(scorelist):
        cutoff = scorelist[BeamWidth - 1]
    else:
        scorelist[-1]
    print(cutoff)
    PrunedPathsWithTerminalSymbol = set()
    for p in PathsWithTerminalSymbol:
        if PathScore[p][0] >= cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    
    PrunedPathsWithTerminalBlank = set()
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p][0] >= cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]

    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    MergedPath = PathsWithTerminalSymbol
    FinalPathScore = PathScore

    for p in PathsWithTerminalBlank:
        if p in MergedPath:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPath.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    print(MergedPath)

    return MergedPath, FinalPathScore




