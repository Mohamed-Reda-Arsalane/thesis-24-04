
from codecs import backslashreplace_errors
from openpyxl import load_workbook
import string
import numpy as np
import pandas as pd
from math import log
import os
import csv
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def concatenate(foo):
    content = ""
    test = foo.split("\n")
    for i in test:
        i = i.removeprefix("[").removeprefix(" ").removeprefix("{'code': \\'").removeprefix("{'code': '").rstrip(
        ).removesuffix("]").removesuffix("\\'}").removesuffix("'}")
        if i.strip() != "":
            content += i + "\\n"
    return content


def concatenate2(foo):
    content = ""
    with open(foo+".txt") as f:
        test = f.read().split("\n")
    for i in test:
        i = i.removeprefix("[").removeprefix(" ").removeprefix("{'code': \\'").removeprefix("{'code': '").rstrip(
        ).removesuffix("]").removesuffix("\\'}").removesuffix("'}")
        if i.strip() != "":
            content += i + "\\n"
    return content


def Addspace(string):
    if "(" in string:
        result = string.split("(")[0]
    else:
        result = string
    return result


def Entropy(stList):
    ent = 0.0
    alphabet = list(set(stList))
    freqList = []
    for symbol in alphabet:
        ctr = 0
        for sym in stList:
            if sym == symbol:
                ctr += 1
        freqList.append(float(ctr) / len(stList))
    for freq in freqList:
        ent = ent + freq * log(freq, 2)
    ent = -ent
    return ent


def Functions(content):
    flen = []
    fnames = []
    fbody = []
    fcall = 0
    fcall_line = ""
    for i in range(0, len(content)):
        if content[i].startswith("Function ") or content[i].startswith("Private Function ") or content[i].startswith("Public Function "):
            count = 0
            fcall_line = i
            temp = ""
            index = content[i].split().index("Function")
            if index != len(content[i].split()) - 1:
                fnames.append(content[i].split()[index+1].split("(")[0])
            while i < len(content) and content[i] != "End Function":
                for j in content[i]:
                    count += 1
                    temp += j
                i += 1
            for j in list(range(0, fcall_line-1)) + list(range(i+1, len(content))):
                for f in fnames:
                    # give comment should only contains code line
                    # add 2 functions find string and find comments takes a line and returns the content of comment or string if found
                    if " "+f+" " in content[j] or " "+f+"(" in content[j] and " "+f+" =" not in content[j] and not content[j].startswith("Function") and not content[j].startswith("Private Function") and not content[j].startswith("Public Function"):
                        fcall += 1
            fbody.append(temp)
            flen.append(count)
    return flen, fnames, fbody, fcall


def Functioncall(content):
    function_call = {"Data_Type_Conv": 0,
                     "Financial_Functions": 0,
                     "Text_Functions": 0,
                     "Test": 0,
                     "Max_op": 0,
                     "Avg_op": 0,
                     "op": 0,
                     "get_functions": 0,
                     "high_imp_functions": 0,
                     "open_functions": 0,
                     "proc_inter_functions": 0,
                     "auto_functions": 0}
    Data_Type_Conv = ["CBOOL", "CBYTE", "CCUR", "CDATE",
                      "CDBL", "CDEC", "CINT", "CLNG", "CSNG", "CSTR", "CVAR"]
    Financial_Functions = ["DDB", "FV", "IPMT", "IRR", "MIRR", "NPER", "NPV", "PMT", "PPMT", "PV", "RATE",
                           "SLN", "SYD"]
    Text_Functions = ["ASC", "CHR", "&", "FORMAT", "INSTR", "INSTRREV", "LCASE", "LEFT", "LEN", "LTRIM", "MID",
                      "REPLACE", "RIGHT", "RTRIM", "SPACE", "SPLIT", "STR", "STRCOMP", "STRCONV", "STRREVERSE",
                      "TRIM" "UCASE", "VAL"]
    test = ["-", "+", "/", "*", "^"]
    op = ["-", "+", "/", "&", "*", "^", "\\", "Mod",
          "&", "And", "Or", "Not", "=", "Is", "Like", "Eqv", "Imp", "Xor"]
    auto_functions = ["Auto_Open ", "Auto_Close", "AutoOpen ", "AutoClose ",
                      "AutoNew  ", "Auto_New ", "AutoExec ", "AutoExit ", "Auto_Exec "]
    get_functions = ["GetObject", "Get_Object"]
    high_imp_functions = ["AutoExec", "AutoOpen", "DocumentOpen", "AutoExit", "AutoClose", "Document_Close", "DocumentBeforeClose", "DocumentChange", "AutoNew", "Document_New", "NewDocument", "Document_Open", "Document_BeforeClose", "Auto_Open", "Workbook_Open", "Workbook_Activate", "Workbook_Deactivate", "Auto_Close",
                          "Workbook_Close", "App_WorkbookOpen", "App_NewWorkbook", "App_WorkbookBeforeClose", "Workbook_BeforeClose", "FileSave", "CloseWithoutSaving", "FileOpen", "FileClose", "FileExit", "Workbook_SheetSelectionChange", "Workbook_BeforeSave", "FileTemplates", "ViewVBCode", "ToolsMacro", "FormatStyle", "OpenMyMacro", "HelpAbout"]
    open_functions = ["Workbook_Open", "WorkbookOpen", "WorkbookClose", "Workbook_Close", "Document_Open", "Document_Close", "DocumentOpen",
                      "DocumentClose", "DocumentBeforeClose", "DocumentChange", "Document_New", "Workbook_Activate", "Workbook_Deactivate", "NewDocument"]
    proc_inter_functions = ["Open", "Write", "Put", "Output", "Binary", "FileCopy", "CopyFile", "Kill", "CreateTextFile", "ADODB.Stream", "WriteText", "SaveToFile", "vbNormal", "vbNormalFocus", "vbHide", "vbMinimizedFocus", "vbMaximizedFocus", "vbNormalNoFocus", "vbMinimizedNoFocus", "Run", "MacScript", "popen", "exec", "noexit", "ExecutionPolicy", "noprofile", "command", "EncodedCommand", "scriptblock", "AuthorizationManager", "Application.Visible", "ShowWindow", "SW_HIDE", "MkDir", "ActiveWorkbook.SaveAs", "Application.AltStartupPath", "CreateObject", "Windows", "FindWindow", "libc.dylib", "dylib", "CreateThread", "VirtualAlloc", "VirtualAllocEx", "RtlMoveMemory", "EnumSystemLanguageGroups", "EnumDateFormats", "URLDownloadToFileA", "Net.WebClient", "DownloadFile", "DownloadString", "SendKeys", "AppActivate", "CallByName", "RegOpenKeyExAs", "RegOpenKeyEx", "RegCloseKey", "RegQueryValueExA", "RegQueryValueEx", "RegRead", "GetVolumeInformationA", "GetVolumeInformation", "popupkiller", "SbieDll.dll",
                            "SandboxieControlWndClass", "currentuser", "Schmidti", "AccessVBOM", "VBAWarnings", "ProtectedView", "DisableAttachementsInPV", "DisableInternetFilesInPV", "DisableUnsafeLocationsInPV", "blockcontentexecutionfrominternet", "VBProject", "VBComponents", "CodeModule", "AddFromString", "Call", "GetObject", "ExecQuery", "GetStringValue", "GetDWORDValue", "DOMDocument", "IXMLDOMElement", "ComputerName", "Domain", "RegRead", "RegWrite", "appdata", "WriteLine", "Cells", "Sleep", "Process", "CommandBars", "setRequestHeader", "Send", "setOption", "RecentFiles", "Mozilla", "UserName", "DeleteFile", "Delete", "Execute", "Content", "MsgBox", "Quit", "Run", "Now", "Comments", "PROCESSOR_ARCHITECTURE", "CopyFolder", "winmgmts", "bin", "base64", "CreateKey", "Create", "SpawnInstance", "Selection", "CreateShortcut", "CreateFolder", "DynamicInvoke", "CreateInstance", "MSFConnect", "RegisterTaskDefinition", "Shell", "Application", "ShellExecute", "WScript", "Load", "transformNode", "ExecuteExcel4Macro", "Show"]

    opratio = 0
    max = 0
    index = 0
    sum = 0
    for i in content:
        count = 0
        countop = 0
        for j in i.split():
            if j.upper() in Data_Type_Conv or Addspace(j).upper() in Data_Type_Conv:
                function_call["Data_Type_Conv"] += 1
            if j.upper() in Financial_Functions or Addspace(j).upper() in Financial_Functions:
                function_call["Financial_Functions"] += 1
            if j.upper() in Text_Functions or Addspace(j).upper() in Text_Functions:
                function_call["Text_Functions"] += 1
            if j in test or Addspace(j) in test:
                function_call["Test"] += 1
                count += 1
            if j in op or Addspace(j) in op:
                function_call["op"] += 1
            if j in get_functions or Addspace(j) in get_functions:
                function_call["get_functions"] += 1
            if j in high_imp_functions or Addspace(j) in high_imp_functions:
                function_call["high_imp_functions"] += 1
            if j in open_functions or Addspace(j) in open_functions:
                function_call["open_functions"] += 1
            if j in proc_inter_functions or Addspace(j) in proc_inter_functions:
                function_call["proc_inter_functions"] += 1
            if j in auto_functions or Addspace(j) in auto_functions:
                function_call["auto_functions"] += 1

        if count != 0:
            index += 1
            sum += count
        if count >= max:
            max = count
    function_call["Max_op"] = max
    if index != 0:
        function_call["Avg_op"] = sum/index
    else:
        function_call["Avg_op"] = -1
    return function_call


def ismacrokeyword(line):
    keywords = ["AutoOpen", "AutoClose", "DocumentOpen",
                "DocumentClose", "Document_open", "Document_close"]
    for k in keywords:
        if k in line:
            return True
    return False


def ishumanreadable(word):
    length = len(word)
    if(len(word) < 15):
        alpha = 0
        vowels = 0
        arr = []
        repetition = True
        for ch in word:
            if ch.isalpha():
                alpha += 1
                if(ch == 'A' or ch == 'a' or ch == 'E' or ch == 'e' or ch == 'I'
                   or ch == 'i' or ch == 'O' or ch == 'o' or ch == 'U' or ch == 'u'):
                    vowels += 1
            arr.append(ch)
        for j in range(0, length-1):
            if 3*arr[j] in word:
                repetition = False
        if alpha/length > 0.7 and 0.2 < vowels/length < 0.6 and repetition:
            return True
        else:
            return False
    else:
        return False


def Isvarshortcut(word):
    Shortcut = {"%", "&", "@",
                "#", "!", "$"}
    Shortcutdict = {"%": "Integer", "&": "Long", "@": "Currency",
                    "#": "Double", "!": "Single", "$": "String"}
    for i in Shortcut:
        if word.endswith(i):
            return Shortcutdict[i]
    return -1

# check if line contains string variable declaration


def isstringdec(line):
    if "\"" not in line:
        return False
    else:
        return True


def Variables(variable_line):
    typedict = {"Numbers": 0,
                "Strings": 0}
    variablestypes = []
    variables = []
    types = []
    lower = 0
    upper = 0
    casing_ratio = 0
    variables_count = 0
    Numbertype = ["Integer", "Long"]
    Stringtype = ["String", "Word.Paragraph"]
    shortcut = 0
    for i in variable_line:
        if i.count(" As ") <= 1:
            word_list = i.split()
            word_list = [word.replace(',', '') for word in word_list]
            if "As" not in word_list:
                del word_list[0]
                for word in word_list:
                    shortcut = Isvarshortcut(word)
                    if shortcut == -1:
                        variablestypes.append(word.replace(  # remove () for byte decleartion
                            '()', '') + ":" + "Object")
                        variables.append(word.replace(
                            '()', ''))
                        types.append("Object")
                        types = list(set(types))
                    else:
                        variablestypes.append(word[:-1] + ":" + shortcut)
                        variables.append(word[:-1])
                        types.append(shortcut)
                        types = list(set(types))
            else:
                if word_list[-2] == "*":  # for cases  ex Dim var As String * 256
                    del word_list[-1]
                    del word_list[-1]
                if word_list[-3] == "As":  # for cases where New is used ex Dim var As New String
                    del word_list[-3]
                del word_list[-2]
                del word_list[0]
                for word in word_list:
                    if word != word_list[-1]:
                        shortcut = Isvarshortcut(word)
                        if shortcut == -1:
                            variablestypes.append(word.replace(
                                '()', '') + ":" + word_list[-1])
                            variables.append(word.replace('()', ''))
                            types.append(word_list[-1])
                            types = list(set(types))
                        else:
                            variablestypes.append(word[:-1] + ":" + shortcut)
                            variables.append(word[:-1])
                            types.append(shortcut)
                            types = list(set(types))
        else:
            # Ex: Dim A As AboveAverage, reda As Integer, w As Adjustments...
            if "," in i:
                for j in i.split(","):
                    if j != i.split(",")[0]:
                        j = "Dim" + j
                    variable_line.append(j)
            # Ex: Dim VJ_(2) As Integer: VJ_(0) = Val("38"): VJ_(1) = Val("72"): Dim INH_ As String...
            elif ":" in i:
                for j in i.split(":"):
                    if "Dim" in j:
                        variable_line.append(j.lstrip())

    for t in types:
        typedict[t] = 0
    for i in Numbertype:
        typedict[i] = 0
    for i in Stringtype:
        typedict[i] = 0

    for i in variablestypes:
        for t in types:
            if i.split(":")[-1] == t:
                typedict[t] += 1

    for i in Numbertype:
        typedict["Numbers"] += typedict[i]

    for i in Stringtype:
        typedict["Strings"] += typedict[i]

    variables_count = len(variables)
    for i in variables:
        for j in i:
            if j.islower():
                lower += 1
            else:
                upper += 1
    if lower != 0:
        casing_ratio = upper/lower
    elif lower == 0 and upper != 0:
        casing_ratio = 200
    elif lower == 0 and upper == 0:
        casing_ratio = 0

    return variables_count, typedict, casing_ratio, variables


def Attributes(Attribute_line):
    attributes = []
    strattributes = []
    for i in Attribute_line:
        if "\"" in i:
            strattributes.append(i[1])
        attributes.append(i[1])
    return strattributes


def Words(content):
    words = []
    symbols = ["~", "`", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "-", "+", "=",
               "{", "}", "[", "]", "|", "\\", ":", ";", "<", ",", ">", ".", "?", "/", "\"", "'"]
    stopsymbols = ["_", "."]
    isymbol = True
    humanreadable = 0
    length = []
    for i in content:
        for s in symbols:
            if s not in stopsymbols:
                i = i.replace(s, " "+s+" ")
        for w in i.split():
            count = 0
            isymbol = True
            for s in stopsymbols:
                if w.endswith(s):
                    w = w[:-1]
                if w.startswith(s):
                    w = w[1:]
            for j in w:
                if j not in symbols:
                    isymbol = False
            if not isymbol and not w.replace(".", "").isdigit():  # and len(w) > 1
                words.append(w)
                for c in w:
                    count += 1
                length.append(count)
                if ishumanreadable(w):
                    humanreadable += 1

    return words, length, humanreadable

# check if string is a URL


def isurl(string):
    if "http" in string or "www." in string and "." in string:
        return True
    else:
        return False


def extract(fo):
    variables_count = 0
    casing_ratio = 0
    int_count = 0
    str_count = 0
    macro_keyword = 0
    max_operation_count = 0
    avg_operation_count = 0
    typedict = {}
    comments = []
    strings = []
    strings = []
    string_text = ""
    contents = []
    comments_count = 0
    char_count = 0
    words = []
    words_lenght = []
    avg_word_length = 0
    var_word_length = 0
    mathchar_count = 0
    mathchar_ratio = 0
    codechar_count = 0
    stringchar_ratio = 0
    string_length = []
    avg_string_length = 0
    commentline_count = 0
    line_count = 0
    chars_inline = []
    char_inline = 0
    avgchar_inline = 0
    bigline = 0
    wordcount_comment = 0
    whitespace = 0
    backslash = 0
    humanreadable = 0
    codeline = []
    comment_word = []
    ent = 0
    upper = 0
    lower = 0
    underline = 0
    URL = 0
    entcomment = 0
    entcode = 0
    listcomments = ""
    listcode = ""
    perc_function_char = 0
    function2chars = 0
    avg_commentline = 0
    test = fo.replace('_\\r', ' ').replace('\\r', '').replace('_\\n', '')
    stList = list(fo.replace('\\r', '').replace('\\n', '\n'))
    ent = Entropy(stList)
    test = test.split("\\n")

    for i in test:
        i = i.lstrip()
        if i.strip() != "":
            char_inline = 0
            i = i.replace("\\'", "'")
            contents.append(i)
            for j in i:
                if j != " ":
                    char_count += 1
                    char_inline += 1
                    if j.isupper():
                        upper += 1
                    if j.islower():
                        lower += 1
                if j == ' ':
                    whitespace += 1
                if j == '_':
                    underline += 1
                if j == '\\':
                    backslash += 1
            chars_inline.append(char_inline)
            if char_inline >= 150:
                bigline += 1
    underlineratio = 0
    if char_count != 0:
        underlineratio = underline/char_count
    casingr = 0
    if lower != 0:
        casingr = upper/lower
    Allwords = Words(contents)
    words = Allwords[0]
    words_lenght = Allwords[1]
    humanreadable = Allwords[2]
#
#
#
#
#
#
#
#
#
#
#
#
    for line in contents:
        if ismacrokeyword(line):
            macro_keyword = 1
        if "\"" in line:
            i = line.split("\"")
            for j in range(0, len(i)):
                if j % 2 != 0:
                    strings.append(i[j])
                    string_text += " " + i[j]
                    if isurl(i[j]):
                        URL += 1

    # for i in strings:
    #    temp = 0
    #    for j in i:
    #        temp += 1
    #    string_length.append(temp)
#
    # if len(string_length) != 0:
    #   avg_string_length = sum(string_length) / len(string_length)
    lene = 0
    strword = []
    avgstrword = 0
    for i in string_text:
        if i != " ":
            lene += 1
    if len(strings) != 0:
        avg_string_length = lene / len(strings)
    string_text = string_text.split()
    # for i in string_text:
    #    strword.append(len(i))
    # if len(strword) != 0:
    #    avgstrword = sum(strword)/len(strword)
    #
    #lene = 0
    # for i in strings:
    #    for j in i.split(" "):
    #        for w in j:
    #            lene += 1
    # if len(strings) != 0:
    #    avgstrword = lene / len(strings)

    for i in contents:
        inline_comment = False
        for j in i.split():  # check if inline comment
            if j.startswith("'") and j not in string_text:
                inline_comment = True
        if i.startswith("'") or inline_comment:
            comments.append(i.split("'")[-1])

        if not inline_comment:
            codeline.append(i)
        else:
            if i.split("'")[0] != "":
                codeline.append(i.split("'")[0])
    for i in comments:
        for j in i:
            comments_count += 1
        for w in i.split():
            if w in words or "'"+w in words:
                wordcount_comment += 1
                comment_word.append(w)
#
    for i in codeline:
        listcode += " " + i
    listcode = list(listcode)
    entcode = Entropy(listcode)
    if comments != []:
        for i in comments:
            listcomments += " " + i
        listcomments = list(listcomments)
        entcomment = Entropy(listcomments)


#
#
#
#
#
#
#
#
#
#
#
    if len(words) != 0:
        humanreadable = humanreadable/len(words)
    wordcount = len(words)
    if len(words_lenght) != 0:
        avg_word_length = sum(words_lenght) / len(words_lenght)
        var_word_length = np.var(words_lenght)
    wordcount_notcomment = len(words)-wordcount_comment
    if len(words) != 0:
        percword_comment = wordcount_comment/len(words)
        percword_notcomment = wordcount_notcomment/len(words)

    variable_line = [line for line in codeline if line.startswith("Dim")]
    Attribute_line = [
        line for line in codeline if line.startswith("Attribute")]

    strattributes = Attributes(Attribute_line)

    fvariables = Variables(variable_line)
    variables_count = fvariables[0]
    typedict = fvariables[1]
    casing_ratio = fvariables[2]
    variables = fvariables[3]
    int_count = typedict["Numbers"]
    str_count = typedict["Strings"]

    if variables_count != 0:
        perc_int = int_count/variables_count
        perc_str = str_count/variables_count
    else:
        perc_int = 0
        perc_str = 0
    if char_count != 0:
        whitespace = whitespace/char_count
        backslash = backslash/char_count
    line_count = len(contents)
    if line_count != 0:
        bigline = bigline/line_count

    if len(chars_inline) != 0:
        avgchar_inline = sum(chars_inline) / len(chars_inline)
    else:
        avgchar_inline = -1

    codechar_count = char_count - comments_count
    if char_count != 0:
        stringchar_ratio = sum(string_length)/char_count
    commentline_count = len(comments)
    stringcount = len(strings)
    if line_count != 0:
        avg_commentline = commentline_count/line_count

    functions = Functions(codeline)
    function_length = functions[0]
    if len(function_length) != 0:
        avg_function_length = sum(function_length) / len(function_length)
    else:
        avg_function_length = 0
    if char_count != 0:
        perc_function_char = sum(function_length) / char_count
        function2chars = len(function_length) / char_count

    funcdict = Functioncall(codeline)
    ftext = funcdict["Text_Functions"]
    ftype = funcdict["Data_Type_Conv"]
    ffinac = funcdict["Financial_Functions"]
    fmath = funcdict["Test"]
    testtype = funcdict["Data_Type_Conv"]
    maxmath = funcdict["Max_op"]
    avgmath = funcdict["Avg_op"]
    op_count = funcdict["op"]
    op_ratio = 0
    ftextw = funcdict["Text_Functions"]
    ftypew = funcdict["Data_Type_Conv"]
    fmathw = funcdict["Test"]
    ffinacw = funcdict["Financial_Functions"]
    fautow = funcdict["auto_functions"]
    fhighw = funcdict["high_imp_functions"]
    fopenw = funcdict["open_functions"]
    fgetw = funcdict["get_functions"]
    fprocw = funcdict["proc_inter_functions"]
    if wordcount != 0:
        op_ratio = op_count/wordcount
    functcall_count = functions[-1] + ftext + ftype + ffinac + fmath

    if functcall_count != 0:
        ftext = ftext/functcall_count
        ftype = ftype/functcall_count
        ffinac = ffinac/functcall_count
        farithm = fmath/functcall_count
    else:
        ftext = -1
        ftype = -1
        ffinac = -1
        farithm = -1

    if wordcount != 0:
        ftextw = ftextw/wordcount
        ftypew = ftypew/wordcount
        farithmw = fmathw/wordcount
        fautow = fautow/wordcount
        fhighw = fhighw/wordcount
        fopenw = fopenw/wordcount
        fgetw = fgetw/wordcount
        fprocw = fprocw/wordcount
    else:
        ftextw = 0
        ftypew = 0
        farithmw = 0
        fautow = 0
        fhighw = 0
        fopenw = 0
        fgetw = 0
        fprocw = 0

    fmath_ratio = 0
    if char_count != 0:
        fmath_ratio = fmath/char_count
    stringchars = 0
    if string_length != []:
        stringchars = sum(string_length)
    percode = 0
    perccomment = 0
    if char_count != 0:
        percode = codechar_count/char_count
        perccomment = comments_count/char_count
    result = []
    test = []
    #print("Number of Variables", variables_count)
    #print("Number of Integer Variables", int_count)
    #print("Number of String Variables", str_count)
    #print("% of Integer Variables", perc_int)
    #print("% of String Variables", perc_str)
    #print("Casing Ratio in Variable Declarations:", casing_ratio)
    # print()
    #print("Number of char", char_count)
    #print("avg number of char per line", avgchar_inline)
    #print("Number of line", line_count)
    #print("% of lines > 150 chars", bigline)
    #print("% whitespace", whitespace)
    #print("% backlash", backslash)
    #print("Ratio of upper case to lower case symbols", casingr)
    #print("Amount of underlines in macro", underline)
    #print("Ratio of underlines to amount of characters", underlineratio)
    #print("Entropy of code characters", entcode)
    #print("Entropy of comments characters", entcomment)
    #print('Shannon entropy:', ent)
    #print("Number of words", wordcount)
    #print("Average length of words", avg_word_length)
    #print("Variance length of words", var_word_length)
    #print("% of human readable words", humanreadable)
    # print()
    #print("Highest Number of Consecutive Mathematical Operations", maxmath)
    #print("Average Number of Consecutive Mathematical Operations", avgmath)
    # print("avg. # of chars per function body", avg_function_length)
    #print("% of chars belonging to a function body", perc_function_char)
    # print("# of function definitions divided by length in characters", function2chars)
    #print("% of type conversion functions called", ftype)
    #print("% of financial functions called", ffinac)
    #print("% of arithmetic functions called", farithm)
    #print("% of text functions called", ftext)
    #print("Appearance frequency of math operation", fmath_ratio)
    #print("Amount of operators in macro", op_count)
    #print("Ratio of operators to words", op_ratio)
    # print()
    #print("% of chars belonging to string", stringchar_ratio)
    #print("Average length of string", avg_string_length)
    #print("Number of Strings", stringcount)
    #print("Average length of string", avg_string_length)
    #print("Number of comment", commentline_count)
    #print("avg. comments per line", avg_commentline)
    #print("Number of char in string", stringchars)
    #print("% of chars not in comments", percode)
    #print("% of chars in comments", perccomment)
    #print("Number of char not in comments", codechar_count)
    #print("Number of chars in comments", comments_count)
    # print()
    #print("Macro Keywords:", macro_keyword)
    #print("Url Count in macro", URL)

    #print("Number of words in comments", wordcount_comment)
    #print("Number of words not in comments", wordcount_notcomment)
    #print("% of words in comments", percword_comment)
    #print("% of words not in comments", percword_notcomment)
    #print("Function calls", functcall_count)
    #print("User Declared Function calls", functions[-1])
    #print("Number of Mathematical Operations", fmath)
    #print("Number of words in comments", wordcount_comment)
    #print("Number of words not in comments", wordcount_notcomment)

    test.append(variables_count)
    test.append(int_count)
    test.append(str_count)
    test.append(perc_int)
    test.append(perc_str)
    test.append(casing_ratio)
    test.append(char_count)
    test.append(avgchar_inline)
    test.append(line_count)
    test.append(bigline)
    test.append(whitespace)
    test.append(backslash)
    test.append(casingr)
    test.append(underline)
    test.append(underlineratio)
    test.append(entcode)
    test.append(entcomment)
    test.append(ent)
    test.append(wordcount)
    test.append(avg_word_length)
    test.append(var_word_length)
    test.append(humanreadable)
    test.append(maxmath)
    test.append(avgmath)
    test.append(avg_function_length)
    test.append(perc_function_char)
    test.append(function2chars)
    test.append(ftype)
    test.append(ffinac)
    test.append(farithm)
    test.append(ftext)
    test.append(fmath_ratio)
    test.append(op_count)
    test.append(op_ratio)
    test.append(ftextw)
    test.append(ftypew)
    test.append(farithmw)
    test.append(fautow)
    test.append(fhighw)
    test.append(fopenw)
    test.append(fgetw)
    test.append(fprocw)
    test.append(stringchar_ratio)
    test.append(avg_string_length)
    test.append(stringcount)
    test.append(commentline_count)
    test.append(avg_commentline)
    test.append(stringchars)
    test.append(percode)
    test.append(perccomment)
    test.append(codechar_count)
    test.append(comments_count)
    test.append(macro_keyword)
    test.append(URL)

    return test


def extract_testcases(fo):
    variables_count = 0
    casing_ratio = 0
    int_count = 0
    str_count = 0
    macro_keyword = 0
    max_operation_count = 0
    avg_operation_count = 0
    typedict = {}
    comments = []
    strings = []
    strings = []
    string_text = ""
    contents = []
    comments_count = 0
    char_count = 0
    words = []
    words_lenght = []
    avg_word_length = 0
    var_word_length = 0
    mathchar_count = 0
    mathchar_ratio = 0
    codechar_count = 0
    stringchar_ratio = 0
    string_length = []
    avg_string_length = 0
    commentline_count = 0
    line_count = 0
    chars_inline = []
    char_inline = 0
    avgchar_inline = 0
    bigline = 0
    wordcount_comment = 0
    whitespace = 0
    backslash = 0
    humanreadable = 0
    codeline = []
    comment_word = []
    ent = 0
    upper = 0
    lower = 0
    underline = 0
    URL = 0
    entcomment = 0
    entcode = 0
    listcomments = ""
    listcode = ""
    test = fo.replace('_\\r', ' ').replace('\\r', '').replace('_\\n', '')
    stList = list(fo.replace('\\r', '').replace('\\n', '\n'))
    ent = Entropy(stList)
    test = test.split("\\n")

    for i in test:
        i = i.lstrip()
        if i.strip() != "":
            char_inline = 0
            i = i.replace("\\'", "'")
            contents.append(i)
            for j in i:
                if j != " ":
                    char_count += 1
                    char_inline += 1
                    if j.isupper():
                        upper += 1
                    if j.islower():
                        lower += 1
                if j == ' ':
                    whitespace += 1
                if j == '_':
                    underline += 1
                if j == '\\':
                    backslash += 1
            chars_inline.append(char_inline)
            if char_inline >= 150:
                bigline += 1
    underlineratio = 0
    if char_count != 0:
        underlineratio = underline/char_count
    casingr = 0
    if lower != 0:
        casingr = upper/lower
    Allwords = Words(contents)
    words = Allwords[0]
    words_lenght = Allwords[1]
    humanreadable = Allwords[2]
    for line in contents:
        if ismacrokeyword(line):
            macro_keyword = 1
        if "\"" in line:
            i = line.split("\"")
            for j in range(0, len(i)):
                if j % 2 != 0:
                    strings.append(i[j])
                    string_text += " " + i[j]
                    if isurl(i[j]):
                        URL += 1
    lene = 0
    strword = []
    avgstrword = 0
    for i in string_text:
        if i != " ":
            lene += 1
    if len(strings) != 0:
        avg_string_length = lene / len(strings)
    string_text = string_text.split()
    for i in contents:
        inline_comment = False
        for j in i.split():  # check if inline comment
            if j.startswith("'") and j not in string_text:
                inline_comment = True
        if i.startswith("'") or inline_comment:
            comments.append(i.split("'")[-1])

        if not inline_comment:
            codeline.append(i)
        else:
            if i.split("'")[0] != "":
                codeline.append(i.split("'")[0])
    for i in comments:
        for j in i:
            comments_count += 1
        for w in i.split():
            if w in words or "'"+w in words:
                wordcount_comment += 1
                comment_word.append(w)
    for i in codeline:
        listcode += " " + i
    listcode = list(listcode)
    entcode = Entropy(listcode)
    if comments != []:
        for i in comments:
            listcomments += " " + i
        listcomments = list(listcomments)
        entcomment = Entropy(listcomments)
    if len(words) != 0:
        humanreadable = humanreadable/len(words)
    wordcount = len(words)
    if len(words_lenght) != 0:
        avg_word_length = sum(words_lenght) / len(words_lenght)
    var_word_length = np.var(words_lenght)
    wordcount_notcomment = len(words)-wordcount_comment
    if len(words) != 0:
        percword_comment = wordcount_comment/len(words)
        percword_notcomment = wordcount_notcomment/len(words)

    variable_line = [line for line in codeline if line.startswith("Dim")]
    Attribute_line = [
        line for line in codeline if line.startswith("Attribute")]

    strattributes = Attributes(Attribute_line)

    fvariables = Variables(variable_line)
    variables_count = fvariables[0]
    typedict = fvariables[1]
    casing_ratio = fvariables[2]
    variables = fvariables[3]
    int_count = typedict["Numbers"]
    str_count = typedict["Strings"]

    if variables_count != 0:
        perc_int = int_count/variables_count
        perc_str = str_count/variables_count
    else:
        perc_int = 0
        perc_str = 0
    if char_count != 0:
        whitespace = whitespace/char_count
        backslash = backslash/char_count
    line_count = len(contents)
    if line_count != 0:
        bigline = bigline/line_count

    if len(chars_inline) != 0:
        avgchar_inline = sum(chars_inline) / len(chars_inline)
    else:
        avgchar_inline = -1

    codechar_count = char_count - comments_count
    if char_count != 0:
        stringchar_ratio = sum(string_length)/char_count
    commentline_count = len(comments)
    stringcount = len(strings)
    if line_count != 0:
        avg_commentline = commentline_count/line_count

    functions = Functions(codeline)
    function_length = functions[0]
    if len(function_length) != 0:
        avg_function_length = sum(function_length) / len(function_length)
    else:
        avg_function_length = 0
    if char_count != 0:
        perc_function_char = sum(function_length) / char_count
        function2chars = len(function_length) / char_count

    funcdict = Functioncall(codeline)
    ftext = funcdict["Text_Functions"]
    ftype = funcdict["Data_Type_Conv"]
    ffinac = funcdict["Financial_Functions"]
    fmath = funcdict["Test"]
    testtype = funcdict["Data_Type_Conv"]
    maxmath = funcdict["Max_op"]
    avgmath = funcdict["Avg_op"]
    op_count = funcdict["op"]
    op_ratio = 0
    ftextw = funcdict["Text_Functions"]
    ftypew = funcdict["Data_Type_Conv"]
    fmathw = funcdict["Test"]
    ffinacw = funcdict["Financial_Functions"]
    fautow = funcdict["auto_functions"]
    fhighw = funcdict["high_imp_functions"]
    fopenw = funcdict["open_functions"]
    fgetw = funcdict["get_functions"]
    fprocw = funcdict["proc_inter_functions"]
    if wordcount != 0:
        op_ratio = op_count/wordcount
    functcall_count = functions[-1] + ftext + ftype + ffinac + fmath

    if functcall_count != 0:
        ftext = ftext/functcall_count
        ftype = ftype/functcall_count
        ffinac = ffinac/functcall_count
        farithm = fmath/functcall_count
    else:
        ftext = -1
        ftype = -1
        ffinac = -1
        farithm = -1

    if wordcount != 0:
        ftextw = ftextw/wordcount
        ftypew = ftypew/wordcount
        farithmw = fmathw/wordcount
        fautow = fautow/wordcount
        fhighw = fhighw/wordcount
        fopenw = fopenw/wordcount
        fgetw = fgetw/wordcount
        fprocw = fprocw/wordcount
    else:
        ftextw = 0
        ftypew = 0
        farithmw = 0
        fautow = 0
        fhighw = 0
        fopenw = 0
        fgetw = 0
        fprocw = 0
    test = []
    test.append(char_count)
    test.append(wordcount)
    test.append(stringcount)
    test.append(wordcount_comment)
    test.append(wordcount_notcomment)
    test.append(line_count)
    test.append(round(ent, 3))
    test.append(round(casingr, 3))
    test.append(round(underline, 3))
    test.append(round(underlineratio, 3))
    test.append(URL)
    test.append(round(entcode, 3))
    test.append(round(entcomment, 3))
    test.append(op_count)
    test.append(round(op_ratio, 3))
    test.append(round(ftextw, 3))
    test.append(round(ftypew, 3))
    test.append(round(farithmw, 3))
    test.append(round(fautow, 3))
    test.append(round(fhighw, 3))
    test.append(round(fopenw, 3))
    test.append(round(fgetw, 3))
    test.append(round(fprocw, 3))
    return test
