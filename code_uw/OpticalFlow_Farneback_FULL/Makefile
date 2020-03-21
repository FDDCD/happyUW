UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  COMPILER  = clang++
endif
ifeq ($(UNAME_S),Linux)
  COMPILER  = icpc
endif

CFLAGS    = -g -MMD -MP -Wall
ifeq "$(shell getconf LONG_BIT)" "64"
  LDFLAGS = `pkg-config --cflags opencv` `pkg-config --libs opencv`
else
  LDFLAGS = `pkg-config --cflags opencv` `pkg-config --libs opencv`
endif
LIBS      =
INCLUDE   =
TARGET    = opencv_exe
SRCDIR    = ./src
ifeq "$(strip $(SRCDIR))" ""
  SRCDIR  = .
endif
SOURCES   = $(wildcard $(SRCDIR)/*.cpp)
OBJDIR    = ./obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR  = .
endif
OBJECTS   = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.cpp=.o)))
DEPENDS   = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $< 

all: clean $(TARGET)

clean:
	-rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)


# main: main.cpp
# 	g++ main.cpp -o main.out `pkg-config --cflags opencv` `pkg-config --libs opencv`
