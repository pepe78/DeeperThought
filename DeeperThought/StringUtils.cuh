#ifndef STRINUTILSCUH
#define STRINUTILSCUH

#include <string>
#include <vector>

using namespace std;

void split_without_space(string &inp, vector<string> &outp, char seprator);
int convertToInt(string &inp);
float convertToFloat(string &inp);
void AppendToFile(string filename, string &text);
string convertToString(int num);
string convertToString(float num);
string getNumbersOnly(string &text);

#endif