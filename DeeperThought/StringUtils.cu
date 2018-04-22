#include "StringUtils.cuh"

#include <fstream>
#include <cstdlib>
#include <sstream>

string convertToString(int num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}

string convertToString(float num)
{
	stringstream ss;
	ss << num;
	return ss.str();
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
	ofs.open(filename.c_str(), std::ofstream::out | std::ofstream::app);
	ofs << text;
	ofs.close();
}
