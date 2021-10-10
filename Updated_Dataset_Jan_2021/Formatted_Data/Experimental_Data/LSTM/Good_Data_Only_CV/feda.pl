for my $i ( 0 .. $#ARGV ) {

    open my $fh, '<', $ARGV[$i]
            or die qq{Unable to open "$ARGV[$i]" for input: $!};

    while ( <$fh> ) {
        chomp;
        my ($first, @rest) = split;
        print $first;
        for my $field ( @rest ) {
            print " *$field $i$field";
        }
        print "\n";
    }
}