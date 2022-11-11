import re
string = "ABAVFCATUFGUgiuqidbidqwdiqu29438278346231e2eqw][][;]..]["
string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
print(string)