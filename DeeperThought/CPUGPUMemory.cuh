#ifndef CPUGPUMEMORY
#define CPUGPUMEMORY

#include <fstream>

using namespace std;

class CPUGPUMemory
{
private:
	void *memCPU;
	void *memGPU;

	size_t size;

	bool is_float;
public:
	CPUGPUMemory(bool _is_float, size_t _size, float _initValues);
	~CPUGPUMemory();

	void Resize(size_t newSize);
	void* GetCPUMemory();
	void* GetGPUMemory();
	int GetSize();

	void CopyCPUtoGPU();
	void CopyGPUtoCPU();

	void SaveToFile(ofstream &os);
	void LoadFromFile(ifstream &is);

	void Reset();
};

#endif
