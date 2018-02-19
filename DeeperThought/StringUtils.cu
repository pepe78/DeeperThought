#include "StringUtils.cuh"

#include <fstream>
#include <string>
#include <cstdlib>

string convertToString(int num)
{
	return to_string(num);
}

string convertToString(float num)
{
	return to_string(num);
}

string getNumbersOnly(string &text)
{
	string ret = "";

	for (size_t i = 0; i < text.size(); i++)
	{
		if (text[i] >= '0' && text[i] <= '9')
		{
			ret += text[i];
		}
	}

	return ret;
}

void split_without_space(string &inp, vector<string> &outp, char seprator)
{
	string tmp = "";
	for (size_t i = 0; i < inp.size(); i++)
	{
		if (inp[i] == ' ')
		{
			continue;
		}
		else if (inp[i] == seprator)
		{
			if (tmp.size() != 0)
			{
				outp.push_back(tmp);
				tmp = "";
			}
		}
		else
		{
			tmp += inp[i];
		}
	}
	if (tmp.size() != 0)
	{
		outp.push_back(tmp);
	}
}

int convertToInt(string &inp)
{
	return atoi(inp.c_str());
}

float convertToFloat(string &inp)
{
	return (float)atof(inp.c_str());
}

void AppendToFile(string filename, string &text)
{
	ofstream ofs;
	ofs.open(filename, std::ofstream::out | std::ofstream::app);
	ofs << text;
	ofs.close();
}