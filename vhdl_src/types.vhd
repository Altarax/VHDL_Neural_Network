--! Import libraries
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;

package types is
    
    --! The frac value for everything
	constant frac : integer := 4;

    --! sfixed array
    subtype sfixed_type is sfixed(15 downto -frac);
	type sfixed_bus_array is array (integer range <>) of sfixed_type;
    
end package types;

package body types is
    
    
    
end package body types;