#ifndef CPUGPUMEMORY
#define CPUGPUMEMORY

class CPUGPUMemory
{
private:
	void *memCPU;
	void *memGPU;

	int size;

	bool is_float;
public:
	CPUGPUMemory(bool _is_float, int _size, float _initValues);
	~CPUGPUMemory();

	void Resize(int newSize);
	void* GetCPUMemory();
	void* GetGPUMemory();
	int GetSize();

	void CopyCPUtoGPU();
	void CopyGPUtoCPU();

	void Reset();
};

#endif
