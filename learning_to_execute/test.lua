require "data"

cmd = torch.CmdLine()
cmd:text()
cmd:text("Generate Data")
cmd:text("Options:")
cmd:option("--start", 0, "starting index")
cmd:option("--n", 1000, "number of files to generate")
cmd:option("--print", false, "print c as generated")
opt = cmd:parse(arg or {})
print(opt)

final = opt.start + opt.n - 1
path = "../data/"

for k = opt.start, final do
  local code, var, output = compose(hardness_fun)
  local input = "#include <stdio.h>\nint main(void){"
  for i = 1, #code do
    input = string.format("%s%s", input, code[i]) 
  end
  input = string.format("%sprintf(\"%s\", %s);return 0;}", input, "%d", var)
  
  local filename = string.format("%s%d.c", path, k)
  local compiled = string.format("%s%d.o", path, k)
  file = torch.DiskFile(filename, "w")
  file:writeString(input)
  file:close()
  os.capture(string.format("gcc -c %s -o %s", filename, compiled))

  if opt.print then print(string.format("%s\n%s\n", filename, input)) end
end